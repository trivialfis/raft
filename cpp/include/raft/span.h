/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once


#include <cuda_runtime.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <rmm/device_uvector.hpp>
#include "cuda_utils.cuh"

#define __ASSERT_STR_HELPER(x) #x
#define CUML_EXPECT(cond, ret)  __builtin_expect((cond), (ret))

#define release_assert(cond)                                          \
  ((cond) ? static_cast<void>(0)                                      \
          : __assert_fail(__ASSERT_STR_HELPER(e), __FILE__, __LINE__, \
                          __PRETTY_FUNCTION__))


namespace raft {
constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();  // NOLINT
/*!
 * The extent E of the span returned by subspan is determined as follows:
 *
 *   - If Count is not dynamic_extent, Count;
 *   - Otherwise, if Extent is not dynamic_extent, Extent - Offset;
 *   - Otherwise, dynamic_extent.
 */
template <std::size_t Extent, std::size_t Offset, std::size_t Count>
struct extent_value : public std::integral_constant<
  std::size_t, Count != dynamic_extent ?
  Count : (Extent != dynamic_extent ? Extent - Offset : Extent)> {};

template <typename T>
struct is_device_span_supported_container : std::false_type {};

template <typename T, typename Alloc>
struct is_device_span_supported_container<thrust::device_vector<T, Alloc>> : std::true_type {};

template <typename T>
struct is_device_span_supported_container<rmm::device_uvector<T>> : std::true_type {};

template <typename T>
struct is_host_span_supported_container : std::false_type {};

template <typename T, typename Alloc>
struct is_host_span_supported_container<std::vector<T, Alloc>> : std::true_type {};

template <typename T, typename Alloc>
struct is_host_span_supported_container<thrust::host_vector<T, Alloc>> : std::true_type {};

template <typename T, std::size_t Extent = dynamic_extent>
class span {
 public:
  using element_type = T;
  using value_type = typename std::remove_cv<T>::type;
  using index_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  using iterator = pointer;
  using const_iterator = const pointer;
  using reverse_iterator = pointer;
  using const_reverse_iterator = const pointer;

 public:
  // ctors
  constexpr span() noexcept = default;
  HDI constexpr span(pointer _ptr, index_type _count)
    : size_(_count), data_(_ptr) {
    release_assert(!(Extent != dynamic_extent && _count != Extent));
    release_assert(_ptr || _count == 0);
  }
  HDI constexpr span(pointer _first, pointer _last)
    : size_(_last - _first), data_(_first) {
    release_assert(data_ || size_ == 0);
  }
  template <std::size_t N>
  HDI explicit constexpr span(element_type (&arr)[N]) noexcept
    : size_(N), data_(&arr[0]) {}

  template <
    typename Container,
    std::enable_if_t<is_device_span_supported_container<Container>::value &&
                     std::is_convertible<
                       std::remove_pointer_t<decltype(thrust::raw_pointer_cast(
                         std::declval<Container&>().data()))> (*)[],
                       T (*)[]>::value>* = nullptr>
  explicit constexpr span(Container& that)
    : span{thrust::raw_pointer_cast(that.data()), that.size()} {}

  template <
    typename Container,
    std::enable_if_t<
      is_host_span_supported_container<Container>::value &&
      std::is_convertible<std::remove_pointer_t<
                            decltype(std::declval<Container&>().data())> (*)[],
                          T (*)[]>::value>* = nullptr>
  explicit constexpr span(Container& that) : span{that.data(), that.size()} {}

  template <
    typename Container,
    std::enable_if_t<
      is_device_span_supported_container<std::remove_cv_t<Container>>::value &&
      std::is_convertible<
        std::remove_pointer_t<decltype(thrust::raw_pointer_cast(
          std::declval<Container&>().data()))> (*)[],
        T (*)[]>::value>* = nullptr>
  explicit constexpr span(Container const& that)
    : span{thrust::raw_pointer_cast(that.data()), that.size()} {}

  template <
    typename Container,
    std::enable_if_t<
      is_host_span_supported_container<std::remove_cv_t<Container>>::value &&
      std::is_convertible<std::remove_pointer_t<
                            decltype(std::declval<Container&>().data())> (*)[],
                          T (*)[]>::value>* = nullptr>
  explicit constexpr span(Container const& that)
    : span{that.data(), that.size()} {}

  // iterators
  HDI auto constexpr begin() const noexcept -> iterator { return {this, 0}; }
  HDI auto constexpr end() const noexcept -> iterator { return {this, size()}; }
  HDI auto constexpr cbegin() const noexcept -> const_iterator {
    return {this, 0};
  }
  HDI auto constexpr cend() const noexcept -> const_iterator {
    return {this, size()};
  }
  HDI auto constexpr rbegin() const noexcept -> reverse_iterator {
    return reverse_iterator{end()};
  }
  HDI auto constexpr rend() const noexcept -> reverse_iterator {
    return reverse_iterator{begin()};
  }
  HDI auto constexpr crbegin() const noexcept -> const_reverse_iterator {
    return const_reverse_iterator{cend()};
  }
  HDI auto constexpr crend() const noexcept -> const_reverse_iterator {
    return const_reverse_iterator{cbegin()};
  }

  // Element access
  HDI auto constexpr front() const -> reference { return (*this)[0]; }
  HDI auto constexpr back() const -> reference { return (*this)[size() - 1]; }
  HDI auto constexpr operator[](index_type _idx) const -> reference {
    release_assert(_idx < size());
    return data()[_idx];
  }
  HDI auto constexpr data() const noexcept -> pointer { return data_; }

  // Observers
  HDI auto constexpr size() const noexcept -> index_type { return size_; }
  HDI auto constexpr size_bytes() const noexcept -> index_type {
    return size() * sizeof(T);
  }
  HDI auto constexpr empty() const noexcept -> bool { return size() == 0; }

  // subviews
  template <std::size_t Offset, std::size_t Count = dynamic_extent>
  HDI auto constexpr subspan() const
    -> span<element_type, extent_value<Extent, Offset, Count>::value> {
    release_assert((Count == dynamic_extent) ? (Offset <= size())
                                             : (Offset + Count <= size()));
    return {data() + Offset, Count == dynamic_extent ? size() - Offset : Count};
  }
  HDI auto constexpr subspan(index_type _offset,
                             index_type _count = dynamic_extent) const
    -> span<element_type, dynamic_extent> {
    release_assert((_count == dynamic_extent) ? (_offset <= size())
                                              : (_offset + _count <= size()));
    return {data() + _offset,
            _count == dynamic_extent ? size() - _offset : _count};
  }

  template <std::size_t Count>
  HDI auto constexpr first() const -> span<element_type, Count> {
    release_assert(Count <= size());
    return {data(), Count};
  }
  HDI auto constexpr first(std::size_t _count) const
    -> span<element_type, dynamic_extent> {
    release_assert(_count <= size());
    return {data(), _count};
  }
  template <std::size_t Count>
  HDI auto constexpr last() const -> span<element_type, Count> {
    release_assert(Count <= size());
    return {data() + size() - Count, Count};
  }
  HDI auto constexpr last(std::size_t _count) const
    -> span<element_type, dynamic_extent> {
    SPAN_CHECK(_count <= size());
    return subspan(size() - _count, _count);
  }

 private:
  T* data_{nullptr};
  std::size_t size_{0};
};
}  // namespace raft
