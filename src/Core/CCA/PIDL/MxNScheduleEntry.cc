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
  : name(n), scht(st), madeSched(false), meta_sema("allMetadataReceived Wait", 0)
{
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
  //NO rank exists for callees from a caller perspective
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

void MxNScheduleEntry::reportMetaRecvDone(int size)
{
  //::std::cout << "Meta " << caller_rep.size() << " of " << size << "\n"; 
  if (size == static_cast<int>(caller_rep.size())) { 
    //One the metadata is here, we don't want anyone
    //to wait for the meta_sema so we raise it 
    //(also perpertually raise it in each redistribution)
    meta_sema.up((int) 4*caller_rep.size());
    //::std::cout << "UP Meta semaphore\n";
  }
}

void MxNScheduleEntry::clear(sched_t sch)
{
  descriptorList::iterator iter;
  if(sch == callee) {
    iter = callee_rep.begin();
    for(;iter != callee_rep.end(); iter++) {
      delete (*iter);
    }
  } else {
    iter = caller_rep.begin();
    for(;iter != caller_rep.end(); iter++) {
      delete (*iter);
    }
  }
  //Delete and reset schedule
  iter = sched.begin();
  for(;iter != sched.end(); iter++) {
    delete (*iter);
  }
  madeSched = false;	
}

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





