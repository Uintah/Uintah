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

struct TagData
{
    const char* name;
    unsigned long long alloc_size;
    unsigned long long high_water;
    unsigned long long num_alloc;
    unsigned long long num_dealloc;
    unsigned long long h0;
    unsigned long long h1;
    unsigned long long h2;
    unsigned long long h3;
};


struct TagBase
{
    virtual TagData data() const = 0;

    template < typename Tag >
    static bool register_tag()
    {
      static Tag t;
      static bool ok = ( s_tags.push_back(&t) , (s_tags.back() == &t) );
      return ok;
    }

    static std::vector<TagBase*> s_tags;
};


struct MallocStats
{
    static FILE* file();

    static bool is_tag_enabled(const TagBase*);

    static bool is_enabled();

  private:

    static const std::vector<std::string> s_malloc_stats;

};


}} // end namespace Uintah::Impl

#define UINTAH_CREATE_TAG(TagName)                                  \
  struct TagName : public ::Uintah::Impl::TagBase                   \
  {                                                                 \
    static constexpr const char* const name()                       \
    { return #TagName; }                                            \
    virtual ::Uintah::Impl::TagData data() const                    \
    {                                                               \
      ::Uintah::Impl::TagData d;                                    \
      d.name = name();                                              \
      d.alloc_size  = ::Lockfree::TagStats<TagName>::alloc_size();  \
      d.high_water  = ::Lockfree::TagStats<TagName>::high_water();  \
      d.num_alloc   = ::Lockfree::TagStats<TagName>::num_alloc();   \
      d.num_dealloc = ::Lockfree::TagStats<TagName>::num_dealloc(); \
      d.h0          = ::Lockfree::TagStats<TagName>::histogram(0);  \
      d.h1          = ::Lockfree::TagStats<TagName>::histogram(1);  \
      d.h2          = ::Lockfree::TagStats<TagName>::histogram(2);  \
      d.h3          = ::Lockfree::TagStats<TagName>::histogram(3);  \
      return d;                                                     \
    }                                                               \
  }

#define UINTAH_REGISTER_TAG(TagName)                                         \
    bool impl_##TagName = ::Uintah::Impl::TagBase::register_tag<TagName>()

namespace Uintah { namespace Tags {

// create default tags first
UINTAH_CREATE_TAG(Global);
UINTAH_CREATE_TAG(MMap);
UINTAH_CREATE_TAG(Malloc);
UINTAH_CREATE_TAG(Pool);

// -------------------------------------

// Remember to register tags in AllocatorTags.cc
// create custom tags here
UINTAH_CREATE_TAG(CommList);
UINTAH_CREATE_TAG(PackedBuffer);
UINTAH_CREATE_TAG(CCVariable);
UINTAH_CREATE_TAG(Array3);
UINTAH_CREATE_TAG(Array3Data);

}} // end namspace Uintah::Tags

namespace Uintah {

template < typename T, typename Tag, template< typename > class BaseAllocator >
using TrackingAllocator = Lockfree::TrackingAllocator< T, Tag, BaseAllocator >;

namespace Impl {

template < typename T >
using MMapAllocator = TrackingAllocator<   T
                                         , Tags::Global
                                         , Lockfree::MMapAllocator
                                       >;

template < typename T >
using MallocAllocator = TrackingAllocator<   T
                                           , Tags::Global
                                           , Lockfree::MallocAllocator
                                         >;
} // end namespace Impl


template < typename T >
using MMapAllocator = TrackingAllocator<   T
                                         , Tags::MMap
                                         , Impl::MMapAllocator
                                       >;

template < typename T >
using MallocAllocator = TrackingAllocator<   T
                                           , Tags::Malloc
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
using PoolAllocator = TrackingAllocator<   T
                                         , Tags::Pool
                                         , Impl::PoolAllocator
                                       >;


//----------------------------------------------------------------------------------
// NOTE:
// for every new Tag, update print_malloc_stats
//----------------------------------------------------------------------------------

// the reduction operations
void print_malloc_stats(MPI_Comm comm, int time_step, int root = 0);

template < typename T > using TagStats = Lockfree::TagStats< T >;

using GlobalStats = Lockfree::TagStats<Tags::Global>;

} // end namespace Uintah

#endif // end CORE_MALLOC_ALLOCATOR_TAGS_HPP
