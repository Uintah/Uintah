/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

} // end namespace

namespace Uintah { namespace Impl {

// declare linkage
std::vector<TagBase*> TagBase::s_tags;

const std::vector<std::string> MallocStats::s_malloc_stats = parse_string(getenv("MALLOC_STATS") != nullptr ? getenv("MALLOC_STATS") : "" );

FILE* MallocStats::file()
{
  if (s_malloc_stats.size() > 0u ) {
    int comm_size = Parallel::getMPISize();
    std::string filename = s_malloc_stats[0] + "." + std::to_string(comm_size);
    static FILE* fp = std::fopen(filename.c_str(), "w");
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
UINTAH_REGISTER_TAG(CCVariable);
UINTAH_REGISTER_TAG(Array3);
UINTAH_REGISTER_TAG(Array3Data);

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

    for (auto && base : Impl::TagBase::s_tags) {
      if (Impl::MallocStats::is_tag_enabled(base)) {
        Impl::TagData d = base->data();
        tag_names.push_back(d.name);
        local_stats.push_back(d.alloc_size);
        local_stats.push_back(d.num_alloc);
        local_stats.push_back(d.num_dealloc);
        local_stats.push_back(d.h0);
        local_stats.push_back(d.h1);
        local_stats.push_back(d.h2);
        local_stats.push_back(d.h3);
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
               , "  %s:\n     total alloc: %s\n      high water: %s\n       #   alloc: %llu\n       # dealloc: %llu\n          <=64 B: %llu\n          <=4 KB: %llu\n         <=32 KB: %llu\n          >32 KB: %llu\n"
               , tag_names[0].c_str()
               , Lockfree::Impl::bytes_to_string(global_stats[0]).c_str()
               , Lockfree::Impl::bytes_to_string(global_high_water[0]).c_str()
               , global_stats[1]
               , global_stats[2]
               , global_stats[3]
               , global_stats[4]
               , global_stats[5]
               , global_stats[6]
             );

      // print tag stats
      for (size_t i = 1, n = tag_names.size(); i < n; ++i) {
        fprintf( p_file
                , "  %s:\n     total alloc: %s\n      high water: %s\n       #   alloc: %llu\n       # dealloc: %llu\n          <=64 B: %llu\n          <=4 KB: %llu\n         <=32 KB: %llu\n          >32 KB: %llu\n"
                , tag_names[i].c_str()
                , Lockfree::Impl::bytes_to_string(global_stats[7*i]).c_str()
                , Lockfree::Impl::bytes_to_string(global_high_water[i]).c_str()
                , global_stats[7*i+1]
                , global_stats[7*i+2]
                , global_stats[7*i+3]
                , global_stats[7*i+4]
                , global_stats[7*i+5]
                , global_stats[7*i+6]
              );
      }

      fprintf(p_file, "  MMap excess: %s\n", Lockfree::Impl::bytes_to_string(Lockfree::Impl::mmap_excess()).c_str());

      fprintf(p_file, "End Timestep: %d\n\n", time_step);
    }
  }
}

} // end namespace Uintah
