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

#include <Core/CCA/PIDL/MxNScheduler.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <Core/CCA/PIDL/MxNMetaSynch.h>
#include <Core/CCA/PIDL/ProxyBase.h>
#include <Core/CCA/Comm/DT/DataTransmitter.h>
#include <iostream>
#include <sstream>
#include <Core/Util/Assert.h>

using namespace SCIRun;   

MxNScheduler::MxNScheduler(sched_t sch)
  :s_mutex("ArrSynch access mutex"),sch_type(sch)
{
}

MxNScheduler::~MxNScheduler()
{
  //clear all remote entries.
  for(schedList::iterator iter = entries.begin();iter != entries.end(); iter++){
    delete ((*iter).second);
  }
  //clear all local entries.
  for(arrRepList::iterator iter = myreps.begin();iter != myreps.end(); iter++){
    delete ((*iter).second);
  }
}

void MxNScheduler::setCalleeRepresentation(std::string distname, MxNArrayRep* arrrep)
{
  //TODO: sychronization needed here

  if(sch_type==caller){
    //this is a client scheduler, so setCalleeRepresentation means
    //setRemoteRepresentation
    schedList::iterator iter = entries.find(distname);  
    if (iter == entries.end()) {
      MxNScheduleEntry* sch_e = new MxNScheduleEntry();
      sch_e->addRep(arrrep);
      entries[distname] = sch_e;
    }
    else {
      ((*iter).second)->addRep(arrrep);
    }
  }
  else{
    arrRepList::iterator iter = myreps.find(distname);  
    if (iter != myreps.end()) {
      //clear the old representaion if it exsits.
      delete iter->second;
    }
    myreps[distname] = arrrep;
  }
}

void MxNScheduler::setCallerRepresentation(std::string distname, ProxyID uuid,
						    MxNArrayRep* arrrep)
{
  //TODO: sychronization needed here
  if(sch_type==callee){
    //this is a server scheduler, so setCallerRepresentation means
    //setRemoteRepresentation
    schedList::iterator iter = entries.find(distname+uuid.str());
    if (iter == entries.end()) {
      MxNScheduleEntry* sch_e = new MxNScheduleEntry();
      sch_e->addRep(arrrep);
      entries[distname+uuid.str()] = sch_e;
    }
    else {
      //addRep will clear existed array representation entry.
      ((*iter).second)->addRep(arrrep);
    }
  }
  else{
    arrRepList::iterator iter = myreps.find(distname+uuid.str());  
    if (iter != myreps.end()) {
      //clear the old representaion if it exsits.
      delete iter->second;
    }
    myreps[distname+uuid.str()] = arrrep;
  }
}

MxNArrayRep* MxNScheduler::callerGetCallerRep(std::string distname, ProxyID uuid)
{
  //return the local representation
  arrRepList::iterator iter = myreps.find(distname+uuid.str());
  if(iter != myreps.end()) {
    return iter->second;
  }
  return NULL;
}

MxNArrayRep* MxNScheduler::calleeGetCalleeRep(std::string distname)
{
  //return the local representation
  arrRepList::iterator iter = myreps.find(distname);
  if(iter != myreps.end()) {
    return iter->second;
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

void MxNScheduler::setArray(std::string distname, ProxyID uuid, int callid, void** arr)
{
  //only callee can call this method.
  ASSERT(sch_type==callee); 

  //first wait for meta data completion. because array synchronization
  //needs to know which proxy will send data.
  //size=0 is dummy size. meta synch should already exist.
  getMetaSynch(distname, uuid, 0)->waitForCompletion();

  ::std::cerr << "MxNSched UUID='" << uuid.str() << "' -- '" << callid << "\n";
  ::std::ostringstream index;

  schedList::iterator sch_iter = entries.find(distname+uuid.str());
  if (sch_iter != entries.end()) {
    s_mutex.lock();
    synchList::iterator sy_iter = ((*sch_iter).second)->s_list.find(callid);
    if (sy_iter != ((*sch_iter).second)->s_list.end()) {
      s_mutex.unlock();
      return ((*sy_iter).second)->setArray(arr);
    }  
    else {

      arrRepList::iterator this_iter = myreps.find(distname);
      //TODO throw exception?
      ASSERT(this_iter!=myreps.end());

      MxNArrSynch* mxnasync = new MxNArrSynch((*sch_iter).second,this_iter->second);

      ((*sch_iter).second)->s_list[callid] = mxnasync;
      s_mutex.unlock();
      return mxnasync->setArray(arr);
    }
  }
}

void* MxNScheduler::waitCompleteInArray(std::string distname, ProxyID uuid, int callid)
{
  //only callee can call this method.
  ASSERT(sch_type==callee); 

  MxNArrSynch* mxnasync=getArrSynch(distname, uuid, callid);
  arrRepList::iterator this_iter = myreps.find(distname);
  if(this_iter==myreps.end()) return NULL;

  //TODO: need remove the array synch entry because it will not be used 
  if(mxnasync!=NULL){
    return mxnasync->waitCompleteInArray(this_iter->second);
  }
  else{ 
    return NULL;
  }
}

void MxNScheduler::waitCompleteOutArray(std::string distname, ProxyID uuid, int callid)
{
  //only callee can call this method.
  ASSERT(sch_type==callee); 
  
  MxNArrSynch* mxnasync=getArrSynch(distname, uuid, callid);
  arrRepList::iterator this_iter = myreps.find(distname);
  if(this_iter==myreps.end()) return;
  
  //TODO: need remove the array synch entry because it will not be used 
  if(mxnasync!=NULL){
    mxnasync->waitCompleteOutArray(this_iter->second);
  }
}


MxNArrSynch* MxNScheduler::getArrSynch(::std::string distname, ::ProxyID uuid, int callid)
{
  //for server scheduler only
  ASSERT(sch_type==callee); 

  //first wait for meta data completion. because array synchronization
  //needs to know which proxy will send data.
  //size=0 is dummy size. meta synch should already exist.
  getMetaSynch(distname, uuid, 0)->waitForCompletion();

  //  ::std::cerr << "MxNSched UUID='" << uuid << "' -- '" << callid << "\n";
  schedList::iterator sch_iter = entries.find(distname+uuid.str());
  if (sch_iter != entries.end()) {
    s_mutex.lock();
    synchList::iterator sy_iter = ((*sch_iter).second)->s_list.find(callid);
    if (sy_iter != ((*sch_iter).second)->s_list.end()) {
      s_mutex.unlock();
      return ((*sy_iter).second);
    }
    else {
      arrRepList::iterator this_iter = myreps.find(distname);
      //TODO throw exception?
      ASSERT(this_iter!=myreps.end());

      MxNArrSynch* mxnasync = new MxNArrSynch((*sch_iter).second,this_iter->second);
      ((*sch_iter).second)->s_list[callid] = mxnasync;
      s_mutex.unlock();
      return mxnasync;
    }  
  }
  return NULL;
}
 

MxNMetaSynch* MxNScheduler::getMetaSynch(::std::string distname, ::ProxyID uuid, int size)
{
  //for server scheduler only
  ASSERT(sch_type==callee); 
  schedList::iterator sch_iter = entries.find(distname+uuid.str());
  if (sch_iter == entries.end()) {
    MxNScheduleEntry* sch_e = new MxNScheduleEntry();
    entries[distname+uuid.str()] = sch_e;
  }
  sch_iter = entries.find(distname+uuid.str());
  s_mutex.lock();
  if(sch_iter->second->meta_sync==NULL){
    sch_iter->second->meta_sync = new MxNMetaSynch(size);
  }
  s_mutex.unlock();
  return sch_iter->second->meta_sync;
}
 


descriptorList* MxNScheduler::getRedistributionReps(std::string distname, ProxyID uuid)
{
  //only client can make the schedule
  ASSERT(sch_type==caller);
  //entries for caller is callee entry, distname is the key
  schedList::iterator iter = entries.find(distname);
  arrRepList::iterator this_iter = myreps.find(distname+uuid.str());
  if (iter != entries.end() && this_iter!=myreps.end())
    return iter->second->makeSchedule( this_iter->second );
  else 
    return NULL;
}

void MxNScheduler::clear(std::string distname, ProxyID uuid, sched_t sch)
{
  if(sch==sch_type){
    //clear the local representation
    arrRepList::iterator found;
    if(sch==caller) found=myreps.find(distname+uuid.str());
    else found=myreps.find(distname);
    if(found!=myreps.end()){
      delete (found->second);
      myreps.erase(found);
    }
  }else{
    //clear the remote entry
    schedList::iterator iter;
    if(sch==caller) iter=entries.find(distname+uuid.str());
    else  iter=entries.find(distname);
    if (iter != entries.end()) {
      ((*iter).second)->clear();
    }
  }
}

void MxNScheduler::print()
{
  std::cerr << "entries.size = " << entries.size() << "\n";
  for(schedList::iterator iter = entries.begin(); iter!= entries.end(); iter++) {
    std::cerr << "!!!!! Printing '" << ((*iter).first) << "'";
    if (sch_type == caller )
      std::cerr << " Caller\n"; 
    else
      std::cerr << " Callee\n"; 
    ((*iter).second)->print(std::cerr);
  }
  std::cerr << "myreps.size = " << myreps.size() << "\n";
  for(arrRepList::iterator iter = myreps.begin(); iter!= myreps.end(); iter++) {
    iter->second->print(std::cerr);
  }
}










