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
  unsigned int i;
  descriptorList::iterator iter;
  for(iter=callee_rep.begin(),i=0; i < callee_rep.size(); i++,iter++) {
    delete (*iter);
  }
  for(iter=caller_rep.begin(),i=0; i < caller_rep.size(); i++,iter++) {
    delete (*iter);
  }
  for(iter=sched.begin(),i=0; i < sched.size(); i++,iter++) {
    delete (*iter);
  }
  callee_rep.clear();
  caller_rep.clear();
  sched.clear();
}

void MxNScheduleEntry::addCallerRep(MxNArrayRep* arr_rep)
{
  if(scht == callee) {
    //Check if a representation with the same rank exists 
    int myrank = arr_rep->getRank();
    descriptorList::iterator iter = caller_rep.begin();
    for(unsigned int i=0; i < caller_rep.size(); i++, iter++) 
    if(myrank == (*iter)->getRank()) { 
      delete (*iter); 
      caller_rep.erase(iter); 
    }
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

descriptorList* MxNScheduleEntry::makeSchedule()
{
  assert(scht == caller);

  if(madeSched) return (&sched);

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
  return (&sched);
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
  unsigned int i;
  descriptorList::iterator iter;
  if(sch == callee) {
    for(i=0, iter=callee_rep.begin(); i < callee_rep.size(); i++,iter++) {
      delete (*iter);
    }
    callee_rep.clear();
  } else {
    for(i=0, iter=caller_rep.begin(); i < caller_rep.size(); i++,iter++) {
      delete (*iter);
    }
    caller_rep.clear();
  }
  //Delete and reset schedule
  for(i=0, iter=sched.begin(); i < sched.size(); i++,iter++) {
    delete (*iter);
  }
  sched.clear();
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





