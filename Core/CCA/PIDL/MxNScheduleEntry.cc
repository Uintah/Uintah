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
 *  MxNScheduleEntry.cc 
 *
 *  Written by:
 *   Kostadin Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002 
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/CCA/PIDL/MxNScheduleEntry.h>
#include <Core/Thread/Thread.h>
#include <assert.h>
using namespace SCIRun;   

MxNScheduleEntry::MxNScheduleEntry(std::string n, sched_t st)
  : name(n), scht(st), recv_sema("getCompleteArray Wait", 0), 
    arr_wait_sema("getArrayWait Semaphore", 0), meta_sema("allMetadataReceived Wait", 0), 
    arr_mutex("Distribution Array Pointer lock"), recv_mutex("Receieved Flag lock"),
    allowArraySet(true), madeSched(false)

{
#ifdef DEBUG_THE_SEMAS
  recv_sema_count=0;
  arr_wait_sema_count=0;
  meta_sema_count=0;
#endif
}

MxNScheduleEntry::~MxNScheduleEntry()
{
  descriptorList::iterator iter;
  iter = callee_rep.begin();
  for(;iter != callee_rep.end(); iter++) {
    delete (*iter);
  }
  iter = caller_rep.begin();
  for(;iter != caller_rep.end(); iter++) {
    delete (*iter);
  }
  iter = sched.begin();
  for(;iter != sched.end(); iter++) {
    delete (*iter);
  }
}

void MxNScheduleEntry::addCallerRep(MxNArrayRep* arr_rep)
{
  if(scht == callee) {
    //Check if a representation with the same rank exists 
    int myrank = arr_rep->getRank();
    descriptorList::iterator iter = caller_rep.begin();
    for(unsigned int i=0; i < caller_rep.size(); i++, iter++) 
    if(myrank == (*iter)->getRank()) delete (*iter);
  }
  caller_rep.push_back(arr_rep);
}

void MxNScheduleEntry::addCalleeRep(MxNArrayRep* arr_rep)
{
  callee_rep.push_back(arr_rep);
}

bool MxNScheduleEntry::isCaller()
{
  return (scht == caller);
}

bool MxNScheduleEntry::isCallee()
{
  return (scht == callee);
}

MxNArrayRep* MxNScheduleEntry::getCallerRep(unsigned int index)
{
  if (index < caller_rep.size())
    return caller_rep[index];
  else
    return NULL;
}

MxNArrayRep* MxNScheduleEntry::getCalleeRep(unsigned int index)
{
  if (index < callee_rep.size())
    return callee_rep[index];
  else
    return NULL;
}

descriptorList MxNScheduleEntry::makeSchedule()
{
  assert(scht == caller);

  if(madeSched) return sched;

  //Create the schedule 
  MxNArrayRep* i_rep;
  MxNArrayRep* this_rep = caller_rep[0];
  for(unsigned int i=0; i < callee_rep.size() ;i++)  {
    i_rep = callee_rep[i];
    assert(i_rep->getDimNum() == this_rep->getDimNum());
    //Determine an intersect
    if(this_rep->isIntersect(i_rep))
      sched.push_back(this_rep->Intersect(i_rep));
    //Set flag to indicate that schedule is now created
    madeSched = true;
  }
  return sched;
}	

void* MxNScheduleEntry::getArray()
{
  //If it is okay to set this array, we return NULL
  //to indicate that we want the array to be set
  if (allowArraySet) 
    return NULL;
  else
    return arr_ptr;
}

void* MxNScheduleEntry::getArrayWait()
{
#ifdef DEBUG_THE_SEMAS
  arr_wait_sema_count--;
#endif
  arr_wait_sema.down();
  allowArraySet = true;
  return arr_ptr;
}

void MxNScheduleEntry::setArray(void** a_ptr)
{
  assert(scht == callee);

  //Make sure all the meta data arrived
  meta_sema.down();
  //set array
  arr_mutex.lock();
  if (allowArraySet) {
    arr_ptr = (*a_ptr);
    allowArraySet = false;
    //Raise getArrayWait's semaphore 
#ifdef DEBUG_THE_SEMAS
    arr_wait_sema_count+= caller_rep.size();
#endif
    arr_wait_sema.up((int) caller_rep.size());
#ifdef DEBUG_THE_SEMAS
    printSemaCounts();
    assert(arr_wait_sema_count >= 0);
#endif
  }
  else { 
    (*a_ptr) = arr_ptr;
  }
  arr_mutex.unlock();
}

void MxNScheduleEntry::setNewArray(void** a_ptr)
{
  assert(scht == callee);

  //Make sure all the meta data arrived
  meta_sema.down();
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

void* MxNScheduleEntry::waitCompleteArray()
{
  descriptorList rl;
 
  assert(scht == callee); 

  //Wait for all meta data communications to finish
  meta_sema.down();
  
  //Determine which redistribution requests we are waiting for
  for(unsigned int i=0; i < caller_rep.size(); i++) {
    if (callee_rep[0]->isIntersect(caller_rep[i]))
      rl.push_back(caller_rep[i]);
  }
  //Return NULL if there is no redistribution
  if (rl.size() == 0) {
    int** temp_ptr = new int*;
    (*temp_ptr) = NULL;
    arr_ptr = (void*) temp_ptr; 
    //Reset variables  
#ifdef DEBUG_THE_SEMAS
    arr_wait_sema_count=0;
#endif
    while (arr_wait_sema.tryDown());
    allowArraySet = true;
    meta_sema.up((int) (2*caller_rep.size()));
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
#ifdef DEBUG_THE_SEMAS
      recv_sema_count--;
#endif
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
#ifdef DEBUG_THE_SEMAS
  arr_wait_sema_count=0;
#endif
  while (arr_wait_sema.tryDown());
  allowArraySet = true;
  meta_sema.up((int) (2*caller_rep.size()));

  return arr_ptr;
}

void MxNScheduleEntry::reportMetaRecvDone(int size)
{
  assert(scht == callee);

  //::std::cout << "Meta " << caller_rep.size() << " of " << size << "\n"; 
  if (size == static_cast<int>(caller_rep.size())) { 
    //One the metadata is here, we don't want anyone
    //to wait for the meta_sema so we raise it 
    //(also perpertually raise it in each redistribution)
    meta_sema.up((int) 4*caller_rep.size());
    //::std::cout << "UP Meta semaphore\n";
  }
}

void MxNScheduleEntry::doReceive(int rank)
{
  assert(scht == callee); 

  //Find the proper arr. representation and mark it is received
  for(unsigned int i=0; i < caller_rep.size(); i++) {
    if (caller_rep[i]->getRank() == rank) {
      //::std::cout << "received dist rank=" << i << "\n";
      //If we have already received this, something is happening
      //out of order, so we yield until things sort out.
      while(caller_rep[i]->received) {
#ifdef DEBUG_THE_SEMA
	recv_sema_count++; 
#endif
	recv_sema.up();
	//Thread::yield();
      }
      recv_mutex.lock();
      caller_rep[i]->received = true;
      recv_mutex.unlock();
#ifdef DEBUG_THE_SEMA
      recv_sema_count++;
#endif
      recv_sema.up();
    }
  }    
}

#ifdef DEBUG_THE_SEMAS
void MxNScheduleEntry::printSemaCounts()
{
  ::std::cout << "__________________________________________________\n";
  ::std::cout << "recv_sema_count = " << recv_sema_count << "\n";
  ::std::cout << "arr_wait_sema_count = " << arr_wait_sema_count << "\n";
  ::std::cout << "meta_sema_count = " << meta_sema_count << "\n";
  ::std::cout << "__________________________________________________\n";
}
#endif

void MxNScheduleEntry::print(std::ostream& dbg)
{
  dbg << "************* Callee Arrays: ***********\n";
  for(unsigned int i=0; i < callee_rep.size(); i++) {
    dbg << "------------------------------\n";
    callee_rep[i]->print(dbg);
  }
  dbg << "************* Caller Arrays: ***********\n";
  for(unsigned int i=0; i < caller_rep.size(); i++) {
    dbg << "------------------------------\n";
    caller_rep[i]->print(dbg);
  }
}





