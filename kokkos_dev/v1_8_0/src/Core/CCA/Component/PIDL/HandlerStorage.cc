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
using namespace SCIRun;

HandlerStorage::HandlerStorage()
  :d_data_sema("Handler Buffer Get Semaphore",0), d_data_mutex("Handler Buffer Map Mutex")
{

}

HandlerStorage::~HandlerStorage()
{
  this->clear();
}

void HandlerStorage::clear(int handler_num)
{
  if (handler_num == 0) {
    /*CLEAR ALL*/
    d_data_mutex.lock();
    /*
    dataList::iterator diter = d_data.begin();
    for(;diter != d_data.end(); diter++) {
      voidvec::iterator viter = (*diter).second.begin();
      for(;viter != (*diter).second.end();viter++) {
	delete (*viter);
      }      
    }
    */
    d_data.clear();
    d_data_mutex.unlock(); 
  }
  else {
    dataList::iterator diter;

    d_data_mutex.lock();
    diter = d_data.find(handler_num);
    if (diter == d_data.end()) {
      d_data_mutex.unlock(); 
      return;
    }
    else {
      /*
      voidvec::iterator viter = (*diter).second.begin();
      for(;viter != (*diter).second.end();viter++) {
	delete (*viter);
      }
      */
      d_data.erase(diter);
    }
    d_data_mutex.unlock();    

  }
}

void HandlerStorage::add(int handler_num, int queue_num, void* data)
{
  d_data_mutex.lock();
  voidvec::iterator viter = (d_data[handler_num]).begin() + queue_num;
  (*(d_data.find(handler_num))).second.insert(viter,data);
  d_data_mutex.unlock();
  
  d_data_sema.up();
}

void* HandlerStorage::get(int handler_num, int queue_num)
{
  void* getData;

  d_data_mutex.lock();
  if (d_data.find(handler_num) == d_data.end())
    getData = NULL;
  else
    getData = (d_data[handler_num])[queue_num];
  d_data_mutex.unlock();
  
  while (getData == NULL) {
    d_data_sema.down();

    d_data_mutex.lock();
    if (d_data.find(handler_num) == d_data.end()) {
      getData = NULL;
    }
    else {
      getData = (d_data[handler_num])[queue_num];
    }
    d_data_mutex.unlock();
  }
  return getData;
}





