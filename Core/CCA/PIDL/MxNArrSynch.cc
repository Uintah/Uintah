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
#include <Core/CCA/PIDL/MxNMetaSynch.h>
#include <assert.h>
using namespace SCIRun;   

MxNArrSynch::MxNArrSynch(MxNScheduleEntry* sched, MxNArrayRep *myrep)
  : _sched(sched), recvCondition("recvCondition"), sendCondition("sendCondition"), 
    arrCondition("arrCondition"), arr_mutex("Distribution Array Pointer lock"),
    recv_mutex("Receieved count lock"), send_mutex("Sent count lock"), allowArraySet(true)
{
  _sched->meta_sync->waitForCompletion();
  expected_count=0;
  recv_count=send_count=0;
  //Determine how many redistribution requests we are waiting for
  for(unsigned int i=0; i < _sched->rep.size(); i++) {
    if (myrep->isIntersect(_sched->rep[i])){
      expected_count++;
    }
  }
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
  std::cerr<<" Calling MxNArrSynch::getArrayWait()" <<std::endl;  
  arr_mutex.lock();
  if(allowArraySet) arrCondition.wait(arr_mutex);
  arr_mutex.unlock();
  std::cerr<<" Calling MxNArrSynch::getArrayWait() DONE" <<std::endl;  
  //  allowArraySet = true;
  return arr_ptr;
}

void MxNArrSynch::setArray(void** a_ptr)
{
  std::cerr<<" Calling MxNArrSynch::setArray()" <<std::endl;

  //set array
  arr_mutex.lock();
  if (allowArraySet) {
    arr_ptr = (*a_ptr);
  }
  else { 
    (*a_ptr) = arr_ptr;
  }
  allowArraySet = false;
  arrCondition.conditionBroadcast();
  arr_mutex.unlock();
}

void MxNArrSynch::setNewArray(void** a_ptr)
{
  std::cerr<<" Calling MxNArrSynch::setNewArray()" <<std::endl;
  //set array
  arr_mutex.lock();
  if (allowArraySet) {
    arr_ptr = (*a_ptr);
    allowArraySet = false;
    arrCondition.conditionBroadcast();
  }
  else { 
    (*a_ptr) = arr_ptr;
  }
  arr_mutex.unlock();
}

void* MxNArrSynch::waitCompleteInArray(MxNArrayRep *myrep)
{
  std::cerr<<" Calling MxNArrSynch::waitCompleteInArray()" <<std::endl;
  descriptorList rl;

  //all meta data communications have finished before the constructor can complete

  if(expected_count==0){
    int** temp_ptr = new int*;
    (*temp_ptr) = NULL;
    arr_ptr = (void*) temp_ptr; 
    //while (arr_wait_sema.tryDown());
    //allowArraySet = true;
    return arr_ptr;
  }

  //Wait until all of those requests are received
  recv_mutex.lock();
  ::std::cerr<<"in waitCompleteArray*** recv_count="<<recv_count<<" expected_count="<<expected_count<<" ***\n";
  if(recv_count<expected_count) recvCondition.wait(recv_mutex);
  recv_mutex.unlock();
  return arr_ptr;
}



void MxNArrSynch::waitCompleteOutArray(MxNArrayRep *myrep)
{
  std::cerr<<" Calling MxNArrSynch::waitCompleteOutArray()" <<std::endl;
  descriptorList rl;

  //all meta data communications have finished before the constructor can complete

  if(expected_count==0) return;

  //Wait until all of those requests are received
  send_mutex.lock();
  ::std::cerr<<"in waitCompleteOutArray*** send_count="<<send_count<<" expected_count="<<expected_count<<" ***\n";
  if(send_count<expected_count) sendCondition.wait(send_mutex);
  send_mutex.unlock();
  return;
}

void MxNArrSynch::doReceive(int rank)
{
  recv_mutex.lock();
  recv_count++;
  if(recv_count>expected_count){
    //TODO: throw an exception here
    ::std::cout << "Error: received too many distribution when I got rank=" << rank << "\n";
  }
  ::std::cerr<<"in doReceive*** recv_count="<<recv_count<<" expected_count="<<expected_count<<" ***\n";
  if(recv_count==expected_count) recvCondition.conditionBroadcast();
  recv_mutex.unlock();
}


 void MxNArrSynch::doSend(int rank)
{
  send_mutex.lock();
  send_count++;
  if(send_count>expected_count){
    //TODO: throw an exception here
    ::std::cout << "Error: received too many distribution when I send rank=" << rank << "\n";
  }
  ::std::cerr<<"in doSend*** send_count="<<send_count<<" expected_count="<<expected_count<<" ***\n";
  if(send_count==expected_count) sendCondition.conditionBroadcast();
  send_mutex.unlock();
}

void MxNArrSynch::print(std::ostream& dbg)
{
}





