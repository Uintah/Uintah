/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  MxNArrSynch.cc 
 *
 *  Written by:
 *   Kostadin Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   August 2003 
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Core/CCA/PIDL/MxNArrSynch.h>
#include <Core/Thread/Thread.h>
#include <assert.h>
using namespace SCIRun;   

MxNArrSynch::MxNArrSynch(MxNScheduleEntry* sched)
  : _sched(sched), recv_sema("getCompleteArray Wait", 0), arr_wait_sema("getArrayWait Semaphore", 0), 
    arr_mutex("Distribution Array Pointer lock"), recv_mutex("Receieved Flag lock"), allowArraySet(true) 
{
}

MxNArrSynch::~MxNArrSynch()
{
}

void* MxNArrSynch::getArray()
{
  //If it is okay to set this array, we return NULL
  //to indicate that we want the array to be set
  if (allowArraySet) 
    return NULL;
  else
    return arr_ptr;
}

void* MxNArrSynch::getArrayWait()
{
  arr_wait_sema.down();
  allowArraySet = true;
  return arr_ptr;
}

void MxNArrSynch::setArray(void** a_ptr)
{
  //Make sure all the meta data arrived
  _sched->meta_sema.down();
  //set array
  arr_mutex.lock();
  if (allowArraySet) {
    arr_ptr = (*a_ptr);
    allowArraySet = false;
    //Raise getArrayWait's semaphore 
    arr_wait_sema.up((int) _sched->caller_rep.size());
  }
  else { 
    (*a_ptr) = arr_ptr;
  }
  arr_mutex.unlock();
}

void MxNArrSynch::setNewArray(void** a_ptr)
{
  //Make sure all the meta data arrived
  _sched->meta_sema.down();
  //set array
  arr_mutex.lock();
  if (allowArraySet) {
    arr_ptr = (*a_ptr);
    allowArraySet = false;
  }
  else { 
    (*a_ptr) = arr_ptr;
  }
  arr_mutex.unlock();
}

void* MxNArrSynch::waitCompleteArray()
{
  descriptorList rl;

  //Wait for all meta data communications to finish
  _sched->meta_sema.down();
  
  //Determine which redistribution requests we are waiting for
  for(unsigned int i=0; i < _sched->caller_rep.size(); i++) {
    if (_sched->callee_rep[0]->isIntersect(_sched->caller_rep[i]))
      rl.push_back(_sched->caller_rep[i]);
  }
  //Return NULL if there is no redistribution
  if (rl.size() == 0) {
    int** temp_ptr = new int*;
    (*temp_ptr) = NULL;
    arr_ptr = (void*) temp_ptr; 
    //Reset variables  
    while (arr_wait_sema.tryDown());
    allowArraySet = true;
    _sched->meta_sema.up((int) (2*_sched->caller_rep.size()));
    //**************
    return arr_ptr;
  }
  
  //Wait until all of those requests are received
  descriptorList::iterator iter;
  for(int i=0; i < (int)rl.size(); iter++, i++) {
    if(i == 0) iter = rl.begin();
    recv_mutex.lock();
    bool recv_flag = (*iter)->received;
    recv_mutex.unlock();
    if (recv_flag == false) {
      //::std::cout << "rank=" << i << " of " << rl.size() << " is not here yet\n";
      recv_sema.down();
      i=-1; //Next loop iteration will increment it to 0
    }
    else {
	//::std::cout <<  "rank=" << i << " of " << rl.size() << " is here\n";
    }
  } 
  //::std::cout << "All redistribution have arrived...\n";
  //Mark all of them as not received to get ready for next time
  iter = rl.begin();
  for(unsigned int i=0; i < rl.size(); iter++, i++) {
    recv_mutex.lock();
    (*iter)->received = false;
    recv_mutex.unlock();
  }
 
  //Reset variables  
  while (arr_wait_sema.tryDown());
  allowArraySet = true;
  _sched->meta_sema.up((int) (2*_sched->caller_rep.size()));

  return arr_ptr;
}

void MxNArrSynch::doReceive(int rank)
{
  //Find the proper arr. representation and mark it is received
  for(unsigned int i=0; i < _sched->caller_rep.size(); i++) {
    if (_sched->caller_rep[i]->getRank() == rank) {
      ::std::cout << "received dist rank=" << i << "\n";
      //If we have already received this, something is happening
      //out of order, so we yield until things sort out.
      while(_sched->caller_rep[i]->received) {
	recv_sema.up();
        ::std::cout << "already have dist rank=" << i << "\n";
	//Thread::yield();
      }
      
      recv_mutex.lock();
      _sched->caller_rep[i]->received = true;
      recv_mutex.unlock();
      recv_sema.up();
    }
  }    
}

void MxNArrSynch::print(std::ostream& dbg)
{
}





