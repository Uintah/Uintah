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
 *  Object.h: Base class for all PIDL distributed objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Component_PIDL_Object_h
#define Component_PIDL_Object_h

namespace SCIRun {
  class MutexPool;
}

namespace PIDL {

class TypeInfo;
class ServerContext;
class Reference;
class URL;

/**************************************
 
CLASS
   Object_interface
   
KEYWORDS
   Object, PIDL
   
DESCRIPTION
   The base class for all PIDL based distributed objects.  It provides
   a single public method - getURL, which returns a URL for that object.
   The _getReference method is also in the public section, but should
   be considered private to the implementation and should not be used
   outside of PIDL or sidl generated stubs.

****************************************/

class Object_interface {
public:
  //////////
  // Destructor
  virtual ~Object_interface();

  //////////
  // Return a URL that can be used to contact this object.
  URL getURL() const;

  //////////
  // Internal method to get a reference (startpoint and
  // vtable base) to this object.
  virtual void _getReference(Reference&, bool copy) const;

  //////////
  // Internal method to increment the reference count for
  // this object.
  void _addReference();

  //////////
  // Internal method to decrement the reference count for
  // this object.
  void _deleteReference();

protected:
  //////////
  // Constructor.  Initializes d_serverContext to null,
  // which means that it is a proxy object.  Calls to
  // intializeServer from derived class constructors will
  // set up d_serverContext appropriately.
  Object_interface();

  //////////
  // Set up d_serverContext with the given type information
  // and the void pointer to be passed back to the generated
  // stub code.  This will get called by each constructor
  // in the entire inheritance tree.  The last one to call
  // will be the most derived class, which is the only
  // one that we care about.
  void initializeServer(const TypeInfo* typeinfo, void* ptr);

private:

  friend class Warehouse;

  //////////
  // The context of the server object.  If this is null,
  // then the object is a proxy and there is no server.
  ServerContext* d_serverContext;

  //////////
  // The reference count for this object.
  int ref_cnt;

  //////////
  // The index of the mutex in the mutex pool.
  int mutex_index;

  //////////
  // Activate the server.  This registers the object with the
  // object wharehouse, and creates the globus endpoint.
  // This is not done until something requests a reference
  // to the object though getURL() or _getReference().
  void activateObject() const;

  //////////
  // Return the singleton Mutex pool shared by all instances
  // of Object_interface.
  static SCIRun::MutexPool* getMutexPool();

  //////////
  // Private copy constructor to make copying impossible.
  Object_interface(const Object_interface&);

  //////////
  // Private assignment operator to make assignment impossible.
  Object_interface& operator=(const Object_interface&);
};

/**************************************
 
CLASS
   Object
   
KEYWORDS
   Handle, PIDL
   
DESCRIPTION
   A pointer to the base object class.  Will somebody be replaced
   with a a smart pointer class.

****************************************/

class Object {
  Object_interface* ptr;
public:
  static const TypeInfo* _getTypeInfo();
  typedef Object_interface interfacetype;
  inline Object()
  {
    ptr=0;
  }
  inline Object(Object_interface* ptr)
    : ptr(ptr)
  {
    if(ptr)
      ptr->_addReference();
  }
  inline ~Object()
  {
    if(ptr)
      ptr->_deleteReference();
  }
  inline Object(const Object& copy)
    : ptr(copy.ptr)
  {
    if(ptr)
      ptr->_addReference();
  }
  inline Object& operator=(const Object& copy)
  {
    if(&copy != this){
      if(ptr)
	ptr->_deleteReference();
      if(copy.ptr)
	copy.ptr->_addReference();
    }
    ptr=copy.ptr;
    return *this;
  }
  inline Object_interface* getPointer() const
  {
    return ptr;
  }
  inline Object_interface* operator->() const
  {
    return ptr;
  }
  inline operator bool() const
  {
    return ptr != 0;
  }
};

} // End namespace PIDL

#endif

