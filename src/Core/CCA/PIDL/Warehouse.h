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

#ifndef CCA_PIDL_Warehouse_h
#define CCA_PIDL_Warehouse_h

#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Mutex.h>
#include <sgi_stl_warnings_off.h>
#include <map>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

class Object;

/**************************************
 
CLASS
   Warehouse
   
KEYWORDS
   Warehouse, PIDL
   
DESCRIPTION
   Internal PIDL class. This is a singleton that holds all activated
   server objects.
****************************************/
  class Warehouse {
  public:

    //////////
    // Lookup an object by name.  name should be parsable
    // as an integer, specifiying the object id.  Returns
    // null of the object is not found.  May throw
    // InvalidReference if name is not parsable.
    Object* lookupObject(const std::string&);

    //////////
    // Lookup an object by the object ID.  Returns null if
    // the object is not found.
    Object* lookupObject(int id);

  protected:

    //////////
    // PIDL needs access to most of these methods.
    friend class PIDL;

    //////////
    // The constructor - only called once.
    Warehouse();

    //////////
    // Destructor
    ~Warehouse();

    //////////
    // The Object base class will register server objects with
    // the warehouse.
    friend class Object;

    //////////
    // Register obj with the warehouse, returning the objects
    // unique identifier.
    int registerObject(Object* obj);

    //////////
    // Register obj with the warehouse with the given unique id, 
    // returning the objects unique identifier.
    int registerObject(int id, Object* obj);

    //////////
    // Unregister the object associated with the object ID.
    // Returns a pointer to the object.
    Object* unregisterObject(int id);

    //////////
    // "Run" the warehouse.  This simply blocks until objects
    // have been removed from the warehouse.
    void run();

  private:
    //////////
    // The lock for the object database and nextID
    SCIRun::Mutex mutex;

    //////////
    // The wait condition for run().  It is signaled when all
    // objects have been removed from the warehouse.
    SCIRun::ConditionVariable condition;

    //////////
    // The object database
    std::map<int, Object*> objects;

    //////////
    // The ID of the next object to be created.
    int nextID;
  };
} // End namespace SCIRun

#endif

