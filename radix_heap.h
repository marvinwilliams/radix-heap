#include <algorithm>
#include <array>
#include <bit>

// AVX512CD provides a vector leading-zero count (VPLZCNTD/Q), which is what
// makes vectorising pull() pay: it reduces the key computation to
// xor -> lzcnt -> sub instead of an 8-deep float-exponent sequence.
#if defined(__AVX512CD__) && defined(__AVX512VL__)
#include <immintrin.h>
#define RADIX_HEAP_AVX512 1
#endif
#include <cassert>
#include <climits>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace radix_heap {
namespace internal {
template <typename T>
inline constexpr std::size_t find_bucket(T x, T last) noexcept {
  return sizeof(T) * 8 -
         std::countl_zero(static_cast<std::make_unsigned_t<T>>(x ^ last));
}

template <typename KeyType, bool IsSigned> class encoder_impl_integer;

template <typename KeyType> class encoder_impl_integer<KeyType, false> {
public:
  typedef KeyType key_type;
  typedef KeyType unsigned_key_type;

  inline static constexpr unsigned_key_type encode(key_type x) { return x; }

  inline static constexpr key_type decode(unsigned_key_type x) { return x; }
};

template <typename KeyType> class encoder_impl_integer<KeyType, true> {
public:
  typedef KeyType key_type;
  typedef typename std::make_unsigned<KeyType>::type unsigned_key_type;

  inline static constexpr unsigned_key_type encode(key_type x) {
    return static_cast<unsigned_key_type>(x) ^
           (unsigned_key_type(1) << unsigned_key_type(
                std::numeric_limits<unsigned_key_type>::digits - 1));
  }

  inline static constexpr key_type decode(unsigned_key_type x) {
    return static_cast<key_type>(
        x ^ (unsigned_key_type(1)
             << (std::numeric_limits<unsigned_key_type>::digits - 1)));
  }
};

template <typename KeyType, typename UnsignedKeyType>
class encoder_impl_decimal {
public:
  typedef KeyType key_type;
  typedef UnsignedKeyType unsigned_key_type;

  inline static constexpr unsigned_key_type encode(key_type x) {
    return raw_cast<key_type, unsigned_key_type>(x) ^
           ((-(raw_cast<key_type, unsigned_key_type>(x) >>
               (std::numeric_limits<unsigned_key_type>::digits - 1))) |
            (unsigned_key_type(1)
             << (std::numeric_limits<unsigned_key_type>::digits - 1)));
  }

  inline static constexpr key_type decode(unsigned_key_type x) {
    return raw_cast<unsigned_key_type, key_type>(
        x ^ (((x >> (std::numeric_limits<unsigned_key_type>::digits - 1)) - 1) |
             (unsigned_key_type(1)
              << (std::numeric_limits<unsigned_key_type>::digits - 1))));
  }

private:
  template <typename T, typename U> union raw_cast {
  public:
    constexpr raw_cast(T t) : t_(t) {}
    operator U() const { return u_; }

  private:
    T t_;
    U u_;
  };
};

template <typename KeyType>
class encoder
    : public encoder_impl_integer<KeyType, std::is_signed<KeyType>::value> {};
template <>
class encoder<float> : public encoder_impl_decimal<float, uint32_t> {};
template <>
class encoder<double> : public encoder_impl_decimal<double, uint64_t> {};
} // namespace internal

template <typename KeyType, typename EncoderType = internal::encoder<KeyType>>
class radix_heap {
public:
  typedef KeyType key_type;
  typedef EncoderType encoder_type;
  typedef typename encoder_type::unsigned_key_type unsigned_key_type;

  void push(key_type key) {
    const unsigned_key_type x = encoder_type::encode(key);
    assert(last_ <= x);
    ++size_;
    const size_t k = internal::find_bucket(x, last_);
    buckets_[k].elems.emplace_back(x);
    buckets_[k].min = std::min(buckets_[k].min, x);
    nonempty_ |= bucket_bit(k);
  }

  key_type top() {
    pull();
    return encoder_type::decode(last_);
  }

  void pop() {
    pull();
    buckets_[0].elems.pop_back();
    --size_;
  }

  size_t size() const { return size_; }

  bool empty() const { return size_ == 0; }

  void clear() {
    size_ = 0;
    last_ = key_type();
    for (auto &b : buckets_) {
      b.elems.clear();
      b.min = std::numeric_limits<unsigned_key_type>::max();
    }
    nonempty_ = 0;
  }

  void swap(radix_heap<KeyType, EncoderType> &a) {
    std::swap(size_, a.size_);
    std::swap(last_, a.last_);
    std::swap(nonempty_, a.nonempty_);
    buckets_.swap(a.buckets_);
  }

private:
  struct Bucket {
    std::vector<unsigned_key_type> elems;
    unsigned_key_type min = std::numeric_limits<unsigned_key_type>::max();
  };
  static constexpr std::size_t num_buckets =
      std::numeric_limits<unsigned_key_type>::digits + 1;

  std::size_t size_{0};
  unsigned_key_type last_{};

  // Bit `k - 1` is set while bucket `k` is non-empty, so `pull()` can locate
  // the next non-empty bucket with a single count-trailing-zeros instead of
  // scanning. Bucket 0 is inspected directly and needs no bit, which is what
  // lets buckets 1..digits fit exactly in one `unsigned_key_type`.
  unsigned_key_type nonempty_{0};

  std::array<Bucket, num_buckets> buckets_;

  // Branchless: for k == 0 the left operand is 0, and the shift amount is
  // masked into range so it stays well-defined.
  static constexpr unsigned_key_type bucket_bit(std::size_t k) noexcept {
    return static_cast<unsigned_key_type>(k != 0)
           << ((k - 1) & (std::numeric_limits<unsigned_key_type>::digits - 1));
  }

#ifdef RADIX_HEAP_AVX512
  // Redistribute the drained bucket with AVX512CD.
  //
  // The mask bit for bucket k falls out of the leading-zero count for free:
  // bit k-1 is (1 << (digits-1)) >> lz, and AVX variable shifts yield 0 once
  // the count reaches the element width -- which is exactly the v == 0 case
  // that belongs in bucket 0 and owns no mask bit. No compare needed.
  //
  // 256-bit lanes are deliberate. 512-bit measured slower on Emerald Rapids:
  // the average source bucket holds about ten elements, so 16 lanes overshoot
  // it and the wide form only adds clock and tail cost.
  unsigned_key_type redistribute_avx512(const unsigned_key_type *d,
                                        std::size_t m, unsigned_key_type last,
                                        unsigned_key_type nonempty) {
    std::size_t j = 0;
#define RADIX_HEAP_PLACE(kk, off)                                              \
  do {                                                                         \
    const std::size_t k_ = static_cast<std::size_t>(kk);                       \
    const unsigned_key_type x_ = d[j + (off)];                                 \
    buckets_[k_].elems.emplace_back(x_);                                       \
    buckets_[k_].min = std::min(buckets_[k_].min, x_);                          \
  } while (0)

    if constexpr (sizeof(unsigned_key_type) == 4) {
      const __m256i vlast = _mm256_set1_epi32(static_cast<int>(last));
      const __m256i topbit = _mm256_set1_epi32(static_cast<int>(0x80000000u));
      const __m256i width = _mm256_set1_epi32(32);
      __m256i macc = _mm256_setzero_si256();
      for (; j + 8 <= m; j += 8) {
        const __m256i v = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + j)),
            vlast);
        const __m256i lz = _mm256_lzcnt_epi32(v);
        const __m256i bw = _mm256_sub_epi32(width, lz);
        macc = _mm256_or_si256(macc, _mm256_srlv_epi32(topbit, lz));
        const __m128i lo = _mm256_castsi256_si128(bw);
        const __m128i hi = _mm256_extracti128_si256(bw, 1);
        RADIX_HEAP_PLACE(static_cast<std::uint32_t>(_mm_cvtsi128_si32(lo)), 0);
        RADIX_HEAP_PLACE(static_cast<std::uint32_t>(_mm_extract_epi32(lo, 1)), 1);
        RADIX_HEAP_PLACE(static_cast<std::uint32_t>(_mm_extract_epi32(lo, 2)), 2);
        RADIX_HEAP_PLACE(static_cast<std::uint32_t>(_mm_extract_epi32(lo, 3)), 3);
        RADIX_HEAP_PLACE(static_cast<std::uint32_t>(_mm_cvtsi128_si32(hi)), 4);
        RADIX_HEAP_PLACE(static_cast<std::uint32_t>(_mm_extract_epi32(hi, 1)), 5);
        RADIX_HEAP_PLACE(static_cast<std::uint32_t>(_mm_extract_epi32(hi, 2)), 6);
        RADIX_HEAP_PLACE(static_cast<std::uint32_t>(_mm_extract_epi32(hi, 3)), 7);
      }
      __m128i o = _mm_or_si128(_mm256_castsi256_si128(macc),
                               _mm256_extracti128_si256(macc, 1));
      o = _mm_or_si128(o, _mm_shuffle_epi32(o, _MM_SHUFFLE(1, 0, 3, 2)));
      o = _mm_or_si128(o, _mm_shuffle_epi32(o, _MM_SHUFFLE(2, 3, 0, 1)));
      nonempty |= static_cast<unsigned_key_type>(_mm_cvtsi128_si32(o));
    } else if constexpr (sizeof(unsigned_key_type) == 8) {
      const __m256i vlast = _mm256_set1_epi64x(static_cast<long long>(last));
      const __m256i topbit =
          _mm256_set1_epi64x(static_cast<long long>(1ull << 63));
      const __m256i width = _mm256_set1_epi64x(64);
      __m256i macc = _mm256_setzero_si256();
      for (; j + 4 <= m; j += 4) {
        const __m256i v = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + j)),
            vlast);
        const __m256i lz = _mm256_lzcnt_epi64(v);
        const __m256i bw = _mm256_sub_epi64(width, lz);
        macc = _mm256_or_si256(macc, _mm256_srlv_epi64(topbit, lz));
        const __m128i lo = _mm256_castsi256_si128(bw);
        const __m128i hi = _mm256_extracti128_si256(bw, 1);
        RADIX_HEAP_PLACE(static_cast<std::uint64_t>(_mm_cvtsi128_si64(lo)), 0);
        RADIX_HEAP_PLACE(static_cast<std::uint64_t>(_mm_extract_epi64(lo, 1)), 1);
        RADIX_HEAP_PLACE(static_cast<std::uint64_t>(_mm_cvtsi128_si64(hi)), 2);
        RADIX_HEAP_PLACE(static_cast<std::uint64_t>(_mm_extract_epi64(hi, 1)), 3);
      }
      __m128i o = _mm_or_si128(_mm256_castsi256_si128(macc),
                               _mm256_extracti128_si256(macc, 1));
      o = _mm_or_si128(o, _mm_unpackhi_epi64(o, o));
      nonempty |= static_cast<unsigned_key_type>(_mm_cvtsi128_si64(o));
    }
    // Any other key width has no vector path: j is still 0 and the scalar
    // loop below redistributes the whole bucket.
#undef RADIX_HEAP_PLACE

    for (; j < m; ++j) {
      const unsigned_key_type x = d[j];
      const size_t k = internal::find_bucket(x, last);
      buckets_[k].elems.emplace_back(x);
      buckets_[k].min = std::min(buckets_[k].min, x);
      nonempty |= bucket_bit(k);
    }
    return nonempty;
  }
#endif // RADIX_HEAP_AVX512

  void pull() {
    assert(size_ > 0);
    if (!buckets_[0].elems.empty())
      return;

    assert(nonempty_ != 0);
    const std::size_t i = std::countr_zero(nonempty_) + 1;
    Bucket &src = buckets_[i];
    const unsigned_key_type last = src.min;
    last_ = last;

    // Keep `last` and the non-empty mask in registers across the loop instead
    // of reading them back through `this` on every element. The instruction
    // count is unchanged -- GCC already hoists them -- but it is worth ~15% on
    // Emerald Rapids for 64-bit keys, which otherwise loses IPC to the
    // read-modify-write of nonempty_.
    unsigned_key_type nonempty = nonempty_ & ~bucket_bit(i);

#ifdef RADIX_HEAP_AVX512
    nonempty = redistribute_avx512(src.elems.data(), src.elems.size(), last,
                                   nonempty);
#else
    for (unsigned_key_type x : src.elems) {
      const size_t k = internal::find_bucket(x, last);
      buckets_[k].elems.emplace_back(x);
      buckets_[k].min = std::min(buckets_[k].min, x);
      nonempty |= bucket_bit(k);
    }
#endif
    nonempty_ = nonempty;

    src.elems.clear();
    src.min = std::numeric_limits<unsigned_key_type>::max();
  }
};

template <typename KeyType, typename ValueType,
          typename EncoderType = internal::encoder<KeyType>>
class pair_radix_heap {
public:
  typedef KeyType key_type;
  typedef ValueType value_type;
  typedef EncoderType encoder_type;
  typedef typename encoder_type::unsigned_key_type unsigned_key_type;

  pair_radix_heap() : size_(0), last_(), buckets_() {
    buckets_min_.fill(std::numeric_limits<unsigned_key_type>::max());
  }

  void push(key_type key, const value_type &value) {
    const unsigned_key_type x = encoder_type::encode(key);
    assert(last_ <= x);
    ++size_;
    const size_t k = internal::find_bucket(x, last_);
    buckets_[k].emplace_back(x, value);
    buckets_min_[k] = std::min(buckets_min_[k], x);
  }

  void push(key_type key, value_type &&value) {
    const unsigned_key_type x = encoder_type::encode(key);
    assert(last_ <= x);
    ++size_;
    const size_t k = internal::find_bucket(x, last_);
    buckets_[k].emplace_back(x, std::move(value));
    buckets_min_[k] = std::min(buckets_min_[k], x);
  }

  template <class... Args> void emplace(key_type key, Args &&...args) {
    const unsigned_key_type x = encoder_type::encode(key);
    assert(last_ <= x);
    ++size_;
    const size_t k = internal::find_bucket(x, last_);
    buckets_[k].emplace_back(std::piecewise_construct, std::forward_as_tuple(x),
                             std::forward_as_tuple(args...));
    buckets_min_[k] = std::min(buckets_min_[k], x);
  }

  key_type top_key() {
    pull();
    return encoder_type::decode(last_);
  }

  value_type &top_value() {
    pull();
    return buckets_[0].back().second;
  }

  void pop() {
    pull();
    buckets_[0].pop_back();
    --size_;
  }

  size_t size() const { return size_; }

  bool empty() const { return size_ == 0; }

  void clear() {
    size_ = 0;
    last_ = key_type();
    for (auto &b : buckets_)
      b.clear();
    buckets_min_.fill(std::numeric_limits<unsigned_key_type>::max());
  }

  void swap(pair_radix_heap<KeyType, ValueType, EncoderType> &a) {
    std::swap(size_, a.size_);
    std::swap(last_, a.last_);
    buckets_.swap(a.buckets_);
    buckets_min_.swap(a.buckets_min_);
  }

private:
  size_t size_;
  unsigned_key_type last_;
  std::array<std::vector<std::pair<unsigned_key_type, value_type>>,
             std::numeric_limits<unsigned_key_type>::digits + 1>
      buckets_;
  std::array<unsigned_key_type,
             std::numeric_limits<unsigned_key_type>::digits + 1>
      buckets_min_;

  void pull() {
    assert(size_ > 0);
    if (!buckets_[0].empty())
      return;

    size_t i;
    for (i = 1; buckets_[i].empty(); ++i)
      ;
    last_ = buckets_min_[i];

    for (size_t j = 0; j < buckets_[i].size(); ++j) {
      const unsigned_key_type x = buckets_[i][j].first;
      const size_t k = internal::find_bucket(x, last_);
      buckets_[k].emplace_back(std::move(buckets_[i][j]));
      buckets_min_[k] = std::min(buckets_min_[k], x);
    }
    buckets_[i].clear();
    buckets_min_[i] = std::numeric_limits<unsigned_key_type>::max();
  }
};
} // namespace radix_heap
