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
