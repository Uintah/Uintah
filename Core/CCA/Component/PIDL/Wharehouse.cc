
/*
 *  Wharehouse.h: A pile of distributed objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/PIDL/Wharehouse.h>
#include <Core/CCA/Component/PIDL/InvalidReference.h>
#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/ServerContext.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <sstream>

using Component::PIDL::Object_interface;
using Component::PIDL::Wharehouse;
using std::map;

Wharehouse::Wharehouse()
    : mutex("PIDL::Wharehouse lock"),
      condition("PIDL::Wharehouse objects>0 condition")
{
    nextID=1;
}

Wharehouse::~Wharehouse()
{
}

void Wharehouse::run()
{
    mutex.lock();
    while(objects.size() > 0)
	condition.wait(mutex);
    mutex.unlock();
}

int Wharehouse::registerObject(Object_interface* object)
{
    mutex.lock();
    int id=nextID++;
    objects[id]=object;
    mutex.unlock();
    return id;
}

Object_interface* Wharehouse::unregisterObject(int id)
{
    mutex.lock();
    map<int, Object_interface*>::iterator iter=objects.find(id);
    if(iter == objects.end())
	throw InternalError("Object not in wharehouse");
    objects.erase(id);
    if(objects.size() == 0)
	condition.conditionSignal();
    mutex.unlock();
    return iter->second;
}

Object_interface* Wharehouse::lookupObject(int id)
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

Object_interface* Wharehouse::lookupObject(const std::string& str)
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

int Wharehouse::approval(char* urlstring, globus_nexus_startpoint_t* sp)
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

