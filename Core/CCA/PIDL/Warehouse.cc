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
 *  Warehouse.h: A pile of distributed objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include "Warehouse.h"
#include <Core/CCA/PIDL/InvalidReference.h>
#include <Core/CCA/PIDL/Object.h>
#include <Core/CCA/PIDL/ServerContext.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <sstream>
#include <string>
using namespace SCIRun;
using namespace std;
using std::map;

Warehouse::Warehouse()
  : mutex("PIDL::Warehouse lock"),
    condition("PIDL::Warehouse objects>0 condition")
{
  nextID=1;
}

Warehouse::~Warehouse()
{
}

void Warehouse::run()
{
  mutex.lock();
  while(objects.size() > 0)
    condition.wait(mutex);
  mutex.unlock();
}

int Warehouse::registerObject(Object* object)
{
  mutex.lock();
  int id=nextID++;
  objects[id]=object;
  mutex.unlock();
  return id;
}


int Warehouse::registerObject(int id, Object* object)
{
  mutex.lock();
  objects[id]=object;
  mutex.unlock();
  return id;
}


Object* Warehouse::unregisterObject(int id)
{
  mutex.lock();
  map<int, Object*>::iterator iter=objects.find(id);
  if(iter == objects.end())
    throw SCIRun::InternalError("Object not in wharehouse");
  objects.erase(id);
  if(objects.size() == 0)
    condition.conditionSignal();
  mutex.unlock();
  return iter->second;
}

Object* Warehouse::lookupObject(int id)
{
  mutex.lock();
  map<int, Object*>::iterator iter=objects.find(id);
  if(iter == objects.end()){
    mutex.unlock();
    return 0;
  } else {
    mutex.unlock();
    return iter->second;
  }
}

Object* Warehouse::lookupObject(const std::string& str)
{
  std::istringstream i(str);
  int objid;
  i >> objid;
  if(!i)
    throw InvalidReference("Cannot parse object ID ("+str+")");
  char x;
  i >> x;
  if(i) // If there are more characters, we have a problem...
    throw InvalidReference("Extra characters after object ID ("+str+")");
  return lookupObject(objid);
}

