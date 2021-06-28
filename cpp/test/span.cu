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
#include <gtest/gtest.h>
#include <rmm/device_uvector.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include <raft/handle.hpp>
#include <raft/span.h>

namespace raft {
TEST(Span, Constructor) {
  raft::handle_t handle;

  // rmm::device_uvector
  rmm::device_uvector<float> uvec(100, handle.get_stream_view());
  span<float> s_uvec{uvec};
  ASSERT_EQ(s_uvec.size(), 100);
  ASSERT_EQ(s_uvec.data(), uvec.data());

  rmm::device_uvector<float> const k_uvec(100, handle.get_stream_view());
  span<float const> k_s_uvec{k_uvec};
  ASSERT_EQ(k_s_uvec.size(), 100);
  ASSERT_EQ(k_s_uvec.data(), k_uvec.data());

  // std::vector
  std::vector<float> vec(100, 0);
  span<float> s_vec{vec};
  ASSERT_EQ(s_vec.size(), 100);
  ASSERT_EQ(s_vec.data(), vec.data());

  std::vector<float> const k_vec(100);
  span<float const> k_s_vec{k_vec};
  ASSERT_EQ(k_s_vec.size(), 100);
  ASSERT_EQ(k_s_vec.data(), k_vec.data());

  // thrust::host_vector
  thrust::host_vector<float> hvec(100, 0);
  span<float> s_hvec(hvec);
  ASSERT_EQ(s_hvec.size(), 100);
  ASSERT_EQ(s_hvec.data(), hvec.data());

  thrust::host_vector<float> const k_hvec(100, 0);
  span<float const> k_s_hvec(k_hvec);
  ASSERT_EQ(s_hvec.size(), 100);
  ASSERT_EQ(k_s_hvec.data(), k_hvec.data());

  // thrust::device_vector
  thrust::device_vector<float> dvec(100, 0);
  span<float> s_dvec(hvec);
  ASSERT_EQ(s_dvec.size(), 100);
  ASSERT_EQ(s_dvec.data(), dvec.data());

  thrust::device_vector<float> const k_dvec(100, 0);
  span<float const> k_s_dvec(k_dvec);
  ASSERT_EQ(s_dvec.size(), 100);
  ASSERT_EQ(k_s_dvec.data(), k_dvec.data());

  // static arr
  float arr[10]{0};
  span<float, 10> s_arr(arr);
  ASSERT_EQ(s_arr.size(), 10);
}
}  // namespace raft
