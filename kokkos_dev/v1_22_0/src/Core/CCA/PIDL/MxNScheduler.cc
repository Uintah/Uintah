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
 *  MxNScheduler.cc 
 *
 *  Written by:
 *   Kostadin Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002 
 *
 *  Copyright (C) 2002 SCI Group
 */

#include "MxNScheduler.h"
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <iostream>
#include <sstream>
using namespace SCIRun;   

MxNScheduler::MxNScheduler()
  : s_mutex("ArrSynch access mutex")
{
}

MxNScheduler::~MxNScheduler()
{
  schedList::iterator iter = entries.begin();
  for(;iter != entries.end(); iter++) {
    delete ((*iter).second);
  }
}

void MxNScheduler::setCalleeRepresentation(std::string distname, 
						    MxNArrayRep* arrrep)
{
  schedList::iterator iter = entries.find(distname);  
  if (iter == entries.end()) {
    MxNScheduleEntry* sch_e = new MxNScheduleEntry(distname,callee);
    sch_e->addCalleeRep(arrrep);
    entries[distname] = sch_e;
  }
  else {
    ((*iter).second)->addCalleeRep(arrrep);
  }
}

void MxNScheduler::setCallerRepresentation(std::string distname, 
						    MxNArrayRep* arrrep)
{
  schedList::iterator iter = entries.find(distname);
  if (iter == entries.end()) {
    MxNScheduleEntry* sch_e = new MxNScheduleEntry(distname,caller);
    sch_e->addCallerRep(arrrep);
    entries[distname] = sch_e;
  }
  else {
    ((*iter).second)->addCallerRep(arrrep);
  }
}

MxNArrayRep* MxNScheduler::callerGetCallerRep(std::string distname)
{
  schedList::iterator iter = entries.find(distname);
  if (iter != entries.end()) {
    if (((*iter).second)->isCaller()) {
      return ((*iter).second)->getCallerRep(0);
    }
  }
  return NULL;
}

MxNArrayRep* MxNScheduler::calleeGetCalleeRep(std::string distname)
{
  schedList::iterator iter = entries.find(distname);
  if (iter != entries.end()) {
    if (((*iter).second)->isCallee()) {
      return ((*iter).second)->getCalleeRep(0);
    }
  }
  return NULL;
}

Index* MxNScheduler::makeBlock(int rank, int size, int length)
{
  int sizePproc;
  int sta, fin;

  ////////////
  // Block size per process is calculated using a ceiling division
  sizePproc = (int)std::ceil(length*1.0 / size*1.0);

  sta = rank * sizePproc;
  fin = std::min(((rank+1) * sizePproc),length);
  return (new Index(sta,fin,1));  
}

Index* MxNScheduler::makeCyclic(int rank, int size, int length)
{
  return (new Index(rank,length,size));  
}

void MxNScheduler::reportMetaRecvDone(std::string distname, int size)
{
  schedList::iterator sch_iter = entries.find(distname);
  if (sch_iter != entries.end()) {
    ((*sch_iter).second)->reportMetaRecvDone(size);
  }
}

void MxNScheduler::setArray(std::string distname, std::string uuid, int callid, void** arr)
{
  ::std::cerr << "MxNSched UUID='" << uuid << "' -- '" << callid << "\n";
  ::std::ostringstream index;
  index << uuid << callid;
  schedList::iterator sch_iter = entries.find(distname);
  if (sch_iter != entries.end()) {
    s_mutex.lock();
    synchList::iterator sy_iter = ((*sch_iter).second)->s_list.find(index.str());
    if (sy_iter != ((*sch_iter).second)->s_list.end()) {
      s_mutex.unlock();
      return ((*sy_iter).second)->setArray(arr);
    }  
    else {
      MxNArrSynch* mxnasync = new MxNArrSynch((*sch_iter).second);
      ((*sch_iter).second)->s_list[index.str()] = mxnasync;
      s_mutex.unlock();
      return mxnasync->setArray(arr);
    }
  }
}

void* MxNScheduler::waitCompleteArray(std::string distname, std::string uuid, int callid)
{
  ::std::cerr << "MxNSched UUID='" << uuid << "' -- '" << callid << "\n";
  ::std::ostringstream index;
  index << uuid << callid;
  schedList::iterator sch_iter = entries.find(distname);
  if (sch_iter != entries.end()) {
    s_mutex.lock();
    synchList::iterator sy_iter = ((*sch_iter).second)->s_list.find(index.str());
    if (sy_iter != ((*sch_iter).second)->s_list.end()) {
      s_mutex.unlock();
      return ((*sy_iter).second)->waitCompleteArray();
      s_mutex.lock();
      delete ((*sy_iter).second);
      s_mutex.unlock();
    }
    else {
      MxNArrSynch* mxnasync = new MxNArrSynch((*sch_iter).second);
      ((*sch_iter).second)->s_list[index.str()] = mxnasync;
      s_mutex.unlock();
      return mxnasync->waitCompleteArray();
      s_mutex.lock();
      delete mxnasync;
      s_mutex.unlock();
    }  
  }
  return NULL;
}

MxNArrSynch* MxNScheduler::getArrSynch(::std::string distname, ::std::string uuid, int callid)
{
  ::std::cerr << "MxNSched UUID='" << uuid << "' -- '" << callid << "\n";
  ::std::ostringstream index;
  index << uuid << callid;
  schedList::iterator sch_iter = entries.find(distname);
  if (sch_iter != entries.end()) {
    s_mutex.lock();
    synchList::iterator sy_iter = ((*sch_iter).second)->s_list.find(index.str());
    if (sy_iter != ((*sch_iter).second)->s_list.end()) {
      s_mutex.unlock();
      return ((*sy_iter).second);
    }
    else {
      MxNArrSynch* mxnasync = new MxNArrSynch((*sch_iter).second);
      ((*sch_iter).second)->s_list[index.str()] = mxnasync;
      s_mutex.unlock();
      return mxnasync;
    }  
  }
  return NULL;
}
 
descriptorList* MxNScheduler::getRedistributionReps(std::string distname)
{
  schedList::iterator iter = entries.find(distname);
  if ((iter != entries.end())&&(iter->second->isCaller())) 
    return iter->second->makeSchedule();
  else 
    return NULL;
}

void MxNScheduler::clear(std::string distname, sched_t sch)
{
  schedList::iterator iter = entries.find(distname);
  if (iter != entries.end()) {
    ((*iter).second)->clear(sch);
  }
}

void MxNScheduler::print()
{
  schedList::iterator iter = entries.begin();   
  std::cerr << "entries.size = " << entries.size() << "\n";
  for(unsigned int i=0; i < entries.size(); i++, iter++) {
    std::cerr << "!!!!! Printing '" << ((*iter).first) << "'";
    if (((*iter).second)->isCaller())
      std::cerr << " Caller\n"; 
    else
      std::cerr << " Callee\n"; 
    ((*iter).second)->print(std::cerr);
  }
}










