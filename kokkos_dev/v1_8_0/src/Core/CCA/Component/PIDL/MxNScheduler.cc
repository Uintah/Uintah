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
#include <Core/CCA/Component/PIDL/MxNArrayRep.h>
#include <iostream>
using namespace SCIRun;   

MxNScheduler::MxNScheduler()
{
}

MxNScheduler::~MxNScheduler()
{
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

void* MxNScheduler::getArray(std::string distname)
{
  schedList::iterator iter = entries.find(distname);
  if (iter != entries.end()) {
    return ((*iter).second)->getArray();
  }
  return NULL;
}

void* MxNScheduler::getArrayWait(std::string distname)
{
  schedList::iterator iter = entries.find(distname);
  if (iter != entries.end()) {
    return ((*iter).second)->getArrayWait();
  }
  return NULL;
}

void MxNScheduler::setArray(std::string distname,void** arr)
{
  schedList::iterator iter = entries.find(distname);
  if (iter != entries.end()) {
    ((*iter).second)->setArray(arr);
  }
}

void* MxNScheduler::getCompleteArray(std::string distname)
{
  schedList::iterator iter = entries.find(distname);
  if (iter != entries.end()) {
    return ((*iter).second)->getCompleteArray();
  }
  return NULL;
}

void MxNScheduler::markReceived(std::string distname, int rank)
{
  schedList::iterator iter = entries.find(distname);
  if (iter != entries.end()) {
    ((*iter).second)->doReceive(rank);
  }
}

void MxNScheduler::reportMetaRecvFinished(std::string distname, int size)
{
  schedList::iterator iter = entries.find(distname);
  if (iter != entries.end()) {
    ((*iter).second)->reportMetaRecvFinished(size);
  }
}
 
descriptorList MxNScheduler::getRedistributionReps(std::string distname)
{
  descriptorList rl;
  MxNArrayRep* callee_rep;
  MxNArrayRep* this_rep;
  
  schedList::iterator iter = entries.find(distname);
  if ((iter != entries.end())&&(((*iter).second)->isCaller())) {
    this_rep = ((*iter).second)->getCallerRep(0);
    for(int i=0; ; i++) {
      callee_rep = ((*iter).second)->getCalleeRep(i);
      if (callee_rep == NULL) {
	return rl;
      }
      else if (callee_rep->getDimNum() != this_rep->getDimNum()) {
	continue;
      }
      else {
	//Determine which redistribution requests need this array
	if (this_rep->isIntersect(callee_rep))
	  rl.push_back(callee_rep);
      }
    }
  }
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
  //  if (!dbg) {
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
  /*
  }
  else {
    schedList::iterator iter = entries.begin();   
    dbg << "entries.size = " << entries.size() << "\n";
    for(unsigned int i=0; i < entries.size(); i++, iter++) {
      dbg << "!!!!! Printing '" << ((*iter).first) << "'";
      if (((*iter).second)->isCaller())
	dbg << " Caller\n"; 
      else
	dbg << " Callee\n"; 
      ((*iter).second)->print(dbg);
    }
  }
  */
}










