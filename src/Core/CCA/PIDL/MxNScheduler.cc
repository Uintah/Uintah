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
  return (new Index(sta,fin-1,1));  
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
 
descriptorList MxNScheduler::getRedistributionReps(std::string distname)
{
  descriptorList rl;
  
  schedList::iterator iter = entries.find(distname);
  if ((iter != entries.end())&&(iter->second->isCaller())) 
    return iter->second->makeSchedule();
  else 
    return rl;
}

void MxNScheduler::clear(std::string distname)
{
  schedList::iterator iter = entries.find(distname);
  if (iter != entries.end()) {
    entries.erase(iter);
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










