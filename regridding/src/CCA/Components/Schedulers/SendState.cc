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


#include <CCA/Components/Schedulers/SendState.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InternalError.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

SendState::SendState()
: d_lock("SendState Lock")
{
}

SendState::~SendState()
{
  d_lock.writeLock();
  for( maptype::iterator iter = sendSubsets.begin(); iter != sendSubsets.end();iter++) {
    if( iter->second->removeReference() ) {
      delete iter->second;
    }
  }
  d_lock.writeUnlock();
}

ParticleSubset*
SendState::find_sendset(int dest, const Patch* patch, int matlIndex,
                        IntVector low, IntVector high, int dwid /* =0 */) const
{
  d_lock.readLock();
  ParticleSubset* ret;
  maptype::const_iterator iter = sendSubsets.find( make_pair( PSPatchMatlGhostRange(patch, matlIndex, low, high, dwid), dest) );

  if( iter == sendSubsets.end() ) {
    ret=NULL;
  } else {
    ret=iter->second;
  }
  d_lock.readUnlock();
  return ret;
}

void
SendState::add_sendset(ParticleSubset* sendset, int dest, const Patch* patch, 
                       int matlIndex, IntVector low, IntVector high, int dwid /*=0*/)
{
  d_lock.writeLock();
  maptype::iterator iter = 
    sendSubsets.find(make_pair(PSPatchMatlGhostRange(patch,matlIndex,low,high,dwid), dest));
  if( iter != sendSubsets.end() ) {
    cout << "sendSubset Already exists for sendset:" << *sendset << " on patch:" << *patch << " matl:" << matlIndex << endl;
    SCI_THROW( InternalError( "sendSubset already exists", __FILE__, __LINE__ ) );
  }
  sendSubsets[make_pair(PSPatchMatlGhostRange(patch, matlIndex, low, high, dwid), dest)]=sendset;
  sendset->addReference();
  d_lock.writeUnlock();
}

void
SendState::reset()
{
  d_lock.writeLock();
  sendSubsets.clear();
  d_lock.writeUnlock();
}

void
SendState::print() 
{
  d_lock.readLock();
  for (maptype::iterator iter = sendSubsets.begin(); iter != sendSubsets.end(); iter++) {
    cout << Parallel::getMPIRank() << ' ' << *(iter->second) << " src/dest: " 
         << iter->first.second << "\n";
  }
  d_lock.readUnlock();
}
