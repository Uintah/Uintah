#include <Core/Malloc/AllocatorTags.hpp>
#include <Core/Lockfree/Lockfree_TrackingAllocator.hpp>
#include <Core/Parallel/Parallel.h>
#include <Core/Thread/Thread.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <cstdio>

namespace {

std::vector<std::string> parse_string(std::string str)
{
  std::replace(str.begin(), str.end(), ',', '\n');
  std::istringstream sin(str);

  std::vector<std::string> results;
  std::string s;
  while (!sin.eof()) {
    sin >> s;
    results.push_back(s);
  }

  return std::move(results);
}

} // namespace

namespace Uintah { namespace Impl {

// declare linkage
std::vector<TagBase*> TagBase::s_tags;

const std::vector<std::string> MallocStats::s_malloc_stats = parse_string(getenv("MALLOC_STATS") != nullptr ? getenv("MALLOC_STATS") : "" );

FILE* MallocStats::file()
{
  if (s_malloc_stats.size() > 0u ) {
    static FILE* fp = std::fopen(s_malloc_stats[0].c_str(), "w");
    return fp;
  }
  return stdout;
}

bool MallocStats::is_tag_enabled(const TagBase* tag)
{
  std::string name = tag->data().name;
  for( size_t i=1, n = s_malloc_stats.size(); i<n; ++i) {
    if (name == s_malloc_stats[i]) {
      return true;
    }
  }

  return false;
}

bool MallocStats::is_enabled()
{
  return s_malloc_stats.size() > 0u;
}

}} // end namespace Uintah::Impl

namespace Uintah { namespace Tags { namespace {

// Add default tags first (printed out in the order registered)
UINTAH_REGISTER_TAG(Global);
UINTAH_REGISTER_TAG(MMap);
UINTAH_REGISTER_TAG(Malloc);
UINTAH_REGISTER_TAG(Pool);

// -------------------------------------

// Add custom tags here (printed out in the order registered)
UINTAH_REGISTER_TAG(CommList);
UINTAH_REGISTER_TAG(PackedBuffer);
UINTAH_REGISTER_TAG(GridVariable);

}}} // end namspace Uintah::Tags

namespace Uintah {


//----------------------------------------------------------------------------------
// NOTE:
// for every new Tag, update print_malloc_stats()
//----------------------------------------------------------------------------------

// the reduction operations
void print_malloc_stats(MPI_Comm comm, int time_step, int root)
{
  if (Impl::MallocStats::is_enabled()) {

    std::vector<unsigned long long> local_stats;
    std::vector<unsigned long long> local_high_water;
    std::vector<std::string> tag_names;

    for (const auto & base : Impl::TagBase::s_tags) {
      if (Impl::MallocStats::is_tag_enabled(base)) {
        Impl::TagData d = base->data();
        tag_names.push_back(d.name);
        local_stats.push_back(d.alloc_size);
        local_stats.push_back(d.num_alloc);
        local_stats.push_back(d.num_dealloc);
        local_high_water.push_back(d.high_water);
      }
    }

    std::vector<unsigned long long> global_stats(local_stats.size());
    std::vector<unsigned long long> global_high_water(local_high_water.size());

    MPI_Reduce(  &local_stats[0], &global_stats[0]
               , static_cast<int>(local_stats.size())
               , MPI_UNSIGNED_LONG_LONG
               , MPI_SUM, root, comm);

    MPI_Reduce(  &local_high_water[0], &global_high_water[0]
               , static_cast<int>(local_high_water.size())
               , MPI_UNSIGNED_LONG_LONG
               , MPI_MAX, root, comm);

    // TODO - add reporting on rank of max high water

    if (Parallel::getMPIRank() == root) {
      FILE* p_file = Impl::MallocStats::file();

      fprintf(p_file, "Timestep: %d\n", time_step);

      // print global stats
      fprintf(   p_file
               , "  %s:\n     total alloc: %s\n      high water: %s\n       #   alloc: %llu\n       # dealloc: %llu\n"
               , tag_names[0].c_str()
               , Lockfree::Impl::bytes_to_string(global_stats[0]).c_str()
               , Lockfree::Impl::bytes_to_string(global_high_water[0]).c_str()
               , global_stats[1]
               , global_stats[2]
             );

      // print tag stats
      for (size_t i = 1, n = tag_names.size(); i < n; ++i) {
        fprintf( p_file
                , "  %s:\n     total alloc: %s\n      high water: %s\n       #   alloc: %llu\n       # dealloc: %llu\n"
                , tag_names[i].c_str()
                , Lockfree::Impl::bytes_to_string(global_stats[3*i]).c_str()
                , Lockfree::Impl::bytes_to_string(global_high_water[i]).c_str()
                , global_stats[3*i+1]
                , global_stats[3*i+2]
              );
      }

      fprintf(p_file, "End Timestep: %d\n\n", time_step);
    }
  }
}

} // end namespace Uintah
