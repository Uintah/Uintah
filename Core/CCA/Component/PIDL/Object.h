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


#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/PIDL/MxNArrayRep.h>
#include <Core/CCA/Component/PIDL/MxNScheduler.h>
#include <Core/CCA/Component/PIDL/HandlerStorage.h>
#include <Core/CCA/Component/PIDL/HandlerGateKeeper.h>
#include <Core/CCA/Component/Comm/EpChannel.h>
#include <Core/CCA/SmartPointer.h>


namespace SCIRun {
  class MutexPool;
}

namespace SCIRun {

class TypeInfo;
class ServerContext;  
class Reference;
class URL;

/**************************************
 
CLASS
   Object
   
KEYWORDS
   Object, PIDL
   
DESCRIPTION
   The base class for all PIDL based distributed objects.  It provides
   a single public method - getURL, which returns a URL for that object.
   The _getReference method is also in the public section, but should
   be considered private to the implementation and should not be used
   outside of PIDL or sidl generated stubs.

****************************************/

class Object {
public:
  typedef CCALib::SmartPointer<Object> pointer;

  //////////
  // Destructor
  virtual ~Object();

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

  //////////
  // External method to increment the reference count for
  // this object.
  void addReference();
  
  //////////
  // External method to decrement the reference count for
  // this object.
  void deleteReference();

  //////////
  // Used to set the distribution of a particular array
  // when this object is the callee  
  void setCalleeDistribution(std::string distname, 
			     MxNArrayRep* arrrep);

  //////////
  // The context of the server object.  If this is null,
  // then the object is a proxy and there is no server.
  ServerContext* d_serverContext;

  //////////
  // Method used to create an array distribution scheduler
  void createScheduler();

  /////////
  // Create a subset of processes to service
  // collective calls.
  void createSubset(int ssize);

protected:
  //////////
  // Constructor.  Initializes d_serverContext to null,
  // which means that it is a proxy object.  Calls to
  // intializeServer from derived class constructors will
  // set up d_serverContext appropriately.
  Object();

  //////////
  // Set up d_serverContext with the given type information
  // and the void pointer to be passed back to the generated
  // stub code.  This will get called by each constructor
  // in the entire inheritance tree.  The last one to call
  // will be the most derived class, which is the only
  // one that we care about.
  void initializeServer(const TypeInfo* typeinfo, void* ptr, EpChannel* epc);

private:

  friend class Warehouse;

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
  // of Object.
  static SCIRun::MutexPool* getMutexPool();

  //////////
  // Private copy constructor to make copying impossible.
  Object(const Object&);

  //////////
  // Private assignment operator to make assignment impossible.
  Object& operator=(const Object&);

};

} // End namespace SCIRun

#endif










