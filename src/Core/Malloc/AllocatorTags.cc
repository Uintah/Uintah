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

const std::vector<std::string> MallocStats::s_malloc_stats = parse_string(getenv("MALLOC_STATS") != nullptr ? getenv("MALLOC_STATS") : "" );

FILE* MallocStats::file()
{
  if (s_malloc_stats.size() > 0u ) {
    static FILE* fp = std::fopen(s_malloc_stats[0].c_str(), "w");
    return fp;
  }
  return stdout;
}

template < typename Tag >
bool MallocStats::is_tag_enabled(const Tag&)
{
  for( size_t i=1, n = s_malloc_stats.size(); i<n; ++i) {
    if (std::string(Tag::name()) == s_malloc_stats[i]) {
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

namespace Uintah {


//----------------------------------------------------------------------------------
// NOTE:
// for every new Tag, update both reduce_malloc_stats and print_global_malloc_stats
//----------------------------------------------------------------------------------

// the reduction operations
void print_malloc_stats(MPI_Comm comm, int root)
{
  if (Impl::MallocStats::is_enabled()) {

    std::vector<unsigned long long> local_stats;
    std::vector<unsigned long long> local_high_water;
    std::vector<std::string> tag_names;

    tag_names.push_back("Global");
    local_stats.push_back(Lockfree::TagStats<void>::alloc_size());
    local_stats.push_back(Lockfree::TagStats<void>::num_alloc());
    local_stats.push_back(Lockfree::TagStats<void>::num_dealloc());
    local_high_water.push_back(Lockfree::TagStats<void>::high_water());

    if ( Impl::MallocStats::is_tag_enabled(CommListTag())) {
      tag_names.push_back(CommListTag::name());
      local_stats.push_back(Lockfree::TagStats<CommListTag>::alloc_size());
      local_stats.push_back(Lockfree::TagStats<CommListTag>::num_alloc());
      local_stats.push_back(Lockfree::TagStats<CommListTag>::num_dealloc());
      local_high_water.push_back(Lockfree::TagStats<CommListTag>::high_water());
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

    if (Parallel::getMPIRank() == root) {
      FILE* p_file = Impl::MallocStats::file();
      for (size_t i = 0, n = tag_names.size(); i < n; ++i) {
        fprintf( p_file
                , "%s: alloc_size{%s}, num_alloc{%llu}, num_dealloc{%llu}, high_water{%s}\n"
                , tag_names[i].c_str()
                , Lockfree::Impl::bytes_to_string(global_stats[3*i]).c_str()
                , global_stats[3*i+1]
                , global_stats[3*i+2]
                , Lockfree::Impl::bytes_to_string(global_high_water[i]).c_str()
              );
      }
    }
  }
}

} // end namespace Uintah
