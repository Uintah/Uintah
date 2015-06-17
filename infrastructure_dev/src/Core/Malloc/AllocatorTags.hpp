#ifndef CORE_MALLOC_ALLOCATOR_TAGS_HPP
#define CORE_MALLOC_ALLOCATOR_TAGS_HPP

#include <Core/Lockfree/Lockfree_MallocAllocator.hpp>
#include <Core/Lockfree/Lockfree_MMapAllocator.hpp>
#include <Core/Lockfree/Lockfree_PoolAllocator.hpp>
#include <Core/Lockfree/Lockfree_TrackingAllocator.hpp>

#include <sci_defs/mpi_defs.h>

#include <vector>
#include <string>

namespace Uintah { namespace Impl {

struct MallocStats
{
    static FILE* file();

    template < typename Tag >
    static bool is_tag_enabled(const Tag&);

    static bool is_enabled();

  private:

    static const std::vector<std::string> s_malloc_stats;

};

}} // end namespace Uintah::Impl

namespace Uintah {

struct GlobalTag
{
    static constexpr const char* const name() { return "Global"; }
};

namespace Impl {

template < typename T >
using MMapAllocator = Lockfree::TrackingAllocator<   T
                                                   , GlobalTag
                                                   , Lockfree::MMapAllocator
                                                 >;

template < typename T >
using MallocAllocator = Lockfree::TrackingAllocator<   T
                                                     , GlobalTag
                                                     , Lockfree::MallocAllocator
                                                   >;
} // end namespace Impl

struct MMapTag
{
    static constexpr const char* const name() { return "MMap"; }
};

struct MallocTag
{
    static constexpr const char* const name() { return "Malloc"; }
};

struct PoolTag
{
    static constexpr const char* const name() { return "Pool"; }
};

template < typename T >
using MMapAllocator = Lockfree::TrackingAllocator<   T
                                                   , MMapTag
                                                   , Impl::MMapAllocator
                                                 >;

template < typename T >
using MallocAllocator = Lockfree::TrackingAllocator<   T
                                                     , MallocTag
                                                     , Impl::MallocAllocator
                                                   >;
namespace Impl {

template < typename T >
using PoolAllocator = Lockfree::PoolAllocator<   T
                                               , Uintah::MMapAllocator
                                               , Uintah::MallocAllocator
                                             >;
} // end Impl namespace

template < typename T >
using PoolAllocator = Lockfree::TrackingAllocator<   T
                                                   , PoolTag
                                                   , Impl::PoolAllocator
                                                 >;

struct CommListTag
{
    static constexpr const char* const name() { return "CommList"; }
};

//----------------------------------------------------------------------------------
// NOTE:
// for every new Tag, update print_malloc_stats
//----------------------------------------------------------------------------------

// the reduction operations
void print_malloc_stats(MPI_Comm comm, int time_step, int root = 0);

template < typename T > using TagStats = Lockfree::TagStats< T >;

using GlobalStats = Lockfree::TagStats<GlobalTag>;

} // end namespace Uintah

#endif // end CORE_MALLOC_ALLOCATOR_TAGS_HPP
