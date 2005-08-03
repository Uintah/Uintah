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
    throw SCIRun::InternalError("Object not in wharehouse", __FILE__, __LINE__);
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

