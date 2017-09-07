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


#include <CCA/Components/Schedulers/SendState.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InternalError.h>

using namespace Uintah;

namespace {

// Tags for each CrowdMonitor
struct send_subsets_tag{};

using send_subsets_monitor = Uintah::CrowdMonitor<send_subsets_tag>;

}


SendState::~SendState()
{
  {
    send_subsets_monitor sends_write_lock{ Uintah::CrowdMonitor<send_subsets_tag>::WRITER };

    for (map_type::iterator iter = sendSubsets.begin(); iter != sendSubsets.end(); iter++) {
      if (iter->second->removeReference()) {
        delete iter->second;
      }
    }
  }
}

ParticleSubset*
SendState::find_sendset(       int         dest
                       , const Patch     * patch
                       ,       int         matlIndex
                       ,       IntVector   low
                       ,       IntVector   high
                       ,       int         dwid /* =0 */
                       ) const
{
  {
    send_subsets_monitor sends_write_lock{ Uintah::CrowdMonitor<send_subsets_tag>::READER };

    ParticleSubset* ret;
    map_type::const_iterator iter = sendSubsets.find(std::make_pair(PSPatchMatlGhostRange(patch, matlIndex, low, high, dwid), dest));

    if (iter == sendSubsets.end()) {
      ret = nullptr;
    } else {
      ret = iter->second;
    }
    return ret;
  }
}

void
SendState::add_sendset(       ParticleSubset * sendset
                      ,       int              dest
                      , const Patch          * patch
                      ,       int              matlIndex
                      ,       IntVector        low
                      ,       IntVector        high
                      ,       int              dwid /*=0*/
                      )
{
  {
    send_subsets_monitor sends_write_lock{ Uintah::CrowdMonitor<send_subsets_tag>::WRITER };

    map_type::iterator iter = sendSubsets.find(std::make_pair(PSPatchMatlGhostRange(patch, matlIndex, low, high, dwid), dest));
    if (iter != sendSubsets.end()) {
      std::cout << "sendSubset already exists for sendset:" << *sendset << " on patch:"
                << *patch << " matl:" << matlIndex << std::endl;
      SCI_THROW(InternalError( "sendSubset already exists", __FILE__, __LINE__ ));
    }
    sendSubsets[std::make_pair(PSPatchMatlGhostRange(patch, matlIndex, low, high, dwid), dest)] = sendset;
    sendset->addReference();
  }
}

void
SendState::reset()
{
  {
    send_subsets_monitor sends_write_lock{ Uintah::CrowdMonitor<send_subsets_tag>::WRITER };

    sendSubsets.clear();
  }
}

void
SendState::print() 
{
  {
    send_subsets_monitor sends_write_lock{ Uintah::CrowdMonitor<send_subsets_tag>::READER };

    for (map_type::iterator iter = sendSubsets.begin(); iter != sendSubsets.end(); iter++) {
      std::cout << Parallel::getMPIRank() << ' ' << *(iter->second) << " src/dest: " << iter->first.second << std::endl;
    }
  }
}
