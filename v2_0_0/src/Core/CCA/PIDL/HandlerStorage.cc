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
 *  HandlerStorage.h: Class which preserves data in between separate
 *                    invocations of the same EP handler.
 *                   
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   Sept 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include "HandlerStorage.h"
#include <iostream>
#include <sstream>
using namespace SCIRun;

HandlerStorage::HandlerStorage()
  :d_data_sema("Handler Buffer Get Semaphore",0), d_data_mutex("Handler Buffer Map Mutex"),threadcount(0)
{
}

HandlerStorage::~HandlerStorage()
{
  this->clear();
}

void HandlerStorage::clear()
{
  /*CLEAR ALL*/
  d_data_mutex.lock();
  d_data.clear();
  d_data_mutex.unlock(); 
}

void HandlerStorage::add(int handler_num, int queue_num, void* data, std::string uuid, int callID, int numCalls)
{
  ::std::ostringstream index;
  index << uuid << callID << handler_num;
  /*Move everyone one space forward, to leave spc#0 for numCalls*/
  int qnum = queue_num + 1;
  
  /*Insert data into queue:*/
  d_data_mutex.lock();
  if ((unsigned int)qnum > d_data[index.str()].size())
    d_data[index.str()].resize(qnum);
  dataList::iterator diter = d_data.find(index.str());
  (*diter).second[qnum] = data;
  /*Insert numcalls in space zero:*/
  int* ncalls = (int *) malloc(sizeof(int));
  (*ncalls) = numCalls; 
  (*diter).second[0] = ncalls;
  d_data_mutex.unlock();
 
  /*Release all (up to M-1) threads that could be stuck:*/ 
  /*Heuristically, give each thread a chance to loop twice,*/
  /*and also provide provision for non-accounted threads, if*/
  /*they exist*/ 
  d_data_sema.up(threadcount*2);
  threadcount=0;
}

void* HandlerStorage::get(int handler_num, int queue_num, std::string uuid, int callID)
{
  void* getData;
  ::std::ostringstream index;
  index << uuid << callID << handler_num;
  /*Queue is misaligned by 1, see add()*/
  int qnum = queue_num + 1;

  /*Retreive data*/
  threadcount++;
  d_data_mutex.lock();
  if(d_data.find(index.str()) == d_data.end())
    getData = NULL;
  else
    getData = (d_data[index.str()])[qnum];
  d_data_mutex.unlock();
  
  while (getData == NULL) {
    /*Wait for some sort of data to arrive:*/
    threadcount++;
    d_data_sema.down();

    /*Check if the package we were waiting on arrived:*/ 
    d_data_mutex.lock();
    if(d_data.find(index.str()) == d_data.end()) {
      getData = NULL;
    }
    else {
      getData = (d_data[index.str()])[qnum];
    }
    d_data_mutex.unlock();
  }

  /*see if needs deallocation*/ 
  d_data_mutex.lock();
  dataList::iterator diter;
  int* ncalls;
  diter = d_data.find(index.str());
  ncalls = (int*) *((*diter).second.begin());
  (*ncalls)--;
  if((*ncalls)==0) d_data.erase(diter);   
  d_data_mutex.unlock();


  return getData;
}





