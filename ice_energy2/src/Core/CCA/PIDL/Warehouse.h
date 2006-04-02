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

