/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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





