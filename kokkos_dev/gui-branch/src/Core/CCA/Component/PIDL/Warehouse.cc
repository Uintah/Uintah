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

#include <Core/CCA/Component/PIDL/Warehouse.h>
#include <Core/CCA/Component/PIDL/InvalidReference.h>
#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/ServerContext.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;

using PIDL::Object_interface;
using PIDL::Warehouse;
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

int Warehouse::registerObject(Object_interface* object)
{
  mutex.lock();
  int id=nextID++;
  objects[id]=object;
  mutex.unlock();
  return id;
}

Object_interface* Warehouse::unregisterObject(int id)
{
  mutex.lock();
  map<int, Object_interface*>::iterator iter=objects.find(id);
  if(iter == objects.end())
    throw SCIRun::InternalError("Object not in wharehouse");
  objects.erase(id);
  if(objects.size() == 0)
    condition.conditionSignal();
  mutex.unlock();
  return iter->second;
}

Object_interface* Warehouse::lookupObject(int id)
{
  mutex.lock();
  map<int, Object_interface*>::iterator iter=objects.find(id);
  if(iter == objects.end()){
    mutex.unlock();
    return 0;
  } else {
    mutex.unlock();
    return iter->second;
  }
}

Object_interface* Warehouse::lookupObject(const std::string& str)
{
  string temp = str;
  std::istringstream i(temp);
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

int Warehouse::approval(char* urlstring, globus_nexus_startpoint_t* sp)
{
  URL url(urlstring);
  Object_interface* obj=lookupObject(url.getSpec());
  if(!obj){
    std::cerr << "Unable to find object: " << urlstring
	      << ", rejecting client (code=1002)\n";
    return 1002;
  }
  if(!obj->d_serverContext){
    std::cerr << "Object is corrupt: " << urlstring
	      << ", rejecting client (code=1003)\n";
    return 1003;
  }
  if(int gerr=globus_nexus_startpoint_bind(sp, &obj->d_serverContext->d_endpoint)){
    std::cerr << "Failed to bind startpoint: " << url.getSpec()
	      << ", rejecting client (code=1004)\n";
    std::cerr << "Globus error code: " << gerr << '\n';
    return 1004;
  }
  /* Increment the reference count for this object. */
  obj->_addReference();
  //std::cerr << "Approved connection to " << urlstring << '\n';
  return GLOBUS_SUCCESS;
}

