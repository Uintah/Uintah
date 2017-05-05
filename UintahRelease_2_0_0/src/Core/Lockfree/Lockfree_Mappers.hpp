/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef LOCKFREE_MAPPERS_HPP
#define LOCKFREE_MAPPERS_HPP

#include <atomic>

namespace Lockfree {

struct ThreadIDMapper
{
  ThreadIDMapper() = default;
  ThreadIDMapper( const ThreadIDMapper & ) = default;
  ThreadIDMapper& operator=( const ThreadIDMapper & ) = default;
  ThreadIDMapper( ThreadIDMapper && ) = default;
  ThreadIDMapper& operator=( ThreadIDMapper && ) = default;

  static size_t tid()
  {
    static constexpr size_t one = 1u;
    static std::atomic<size_t> count{0};
    const static thread_local size_t id = count.fetch_add( one, std::memory_order_relaxed );
    return id;
  }

  template <typename... Args>
  size_t operator()(size_t size, const Args&... /*emplace_args*/) const
  {
    return tid() % size;
  }

  size_t operator()(size_t size) const
  {
    return tid() % size;
  }
};

struct CyclicMapper
{
  CyclicMapper() = default;
  CyclicMapper( const CyclicMapper & ) = default;
  CyclicMapper& operator=( const CyclicMapper & ) = default;
  CyclicMapper( CyclicMapper && ) = default;
  CyclicMapper& operator=( CyclicMapper && ) = default;

  static constexpr size_t one = 1u;

  template <typename... Args>
  size_t operator()(size_t size, const Args&... /*emplace_args*/) const
  {
    return m_count.fetch_add( one, std::memory_order_relaxed ) % size;
  }

  template <typename... Args>
  size_t operator()(size_t size) const
  {
    return m_count.fetch_add( one, std::memory_order_relaxed ) % size;
  }
  mutable std::atomic<size_t> m_count{0};
};


} // namespace Lockfree

#endif //LOCKFREE_MAPPERS_HPP
