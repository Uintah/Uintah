#ifndef CORE_MALLOC_ALLOCATOR_TAGS_HPP
#define CORE_MALLOC_ALLOCATOR_TAGS_HPP

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

struct CommListTag
{
    static constexpr const char* const name() { return "CommList"; }
};

//----------------------------------------------------------------------------------
// NOTE:
// for every new Tag, update print_malloc_stats
//----------------------------------------------------------------------------------

// the reduction operations
void print_malloc_stats(MPI_Comm comm, int root = 0);

template < typename T > using TagStats = Lockfree::TagStats<T>;


} // end namespace Uintah

#endif // end CORE_MALLOC_ALLOCATOR_TAGS_HPP
