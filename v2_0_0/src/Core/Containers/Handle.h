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
 *  Handle.h: Smart Pointers..
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Containers_Handle_h
#define SCI_Containers_Handle_h 1

#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>

namespace SCIRun {


/**************************************

 CLASS
 Handle

 KEYWORDS
 Handle

 DESCRIPTION

 Handle.h: Smart Pointers..

 Written by:
 Steven G. Parker
 Department of Computer Science
 University of Utah
 March 1994

 Copyright (C) 1994 SCI Group

 PATTERNS

 WARNING
  
****************************************/

template<class T> class Handle;
template<class T> void Pio(Piostream& stream, Handle<T>& data);

template<class T> class Handle {
  T* rep;
public:

  typedef T   value_type;
  typedef T * pointer_type;

  //////////
  //Create the handle, initializing with a null pointer.
  Handle();

  //////////
  //Attach the handle to a pointer
  Handle(T*);
    
  //////////
  //Copy a handle from another handle
  Handle(const Handle<T>&);

  //////////
  //Assign a handle from another handle.
  Handle<T>& operator=(const Handle<T>&);

  //////////
  //Assign a handle from a pointer.
  Handle<T>& operator=(T*);

  bool Handle<T>::operator==(const Handle<T>& crep) const;
  bool Handle<T>::operator!=(const Handle<T>& crep) const;

  //////////
  //Destroy the handle
  ~Handle();

  void detach();

  inline T* operator->() const {
    ASSERT(rep != 0);
    return rep;
  }

  inline T* get_rep() const {
    return rep;
  }

};

} // End namespace SCIRun

////////////////////////////////////////////////////////////
// Start of included Handle.cc

namespace SCIRun {

template<class T>
Handle<T>::Handle()
: rep(0)
{
}

template<class T>
Handle<T>::Handle(T* rep)
: rep(rep)
{
    if(rep)rep->ref_cnt++;
}

template<class T>
Handle<T>::Handle(const Handle<T>& copy)
: rep(copy.rep)
{
    if(rep)rep->ref_cnt++;
}

template<class T>
Handle<T>& Handle<T>::operator=(const Handle<T>& copy)
{
    if(rep != copy.rep){
	if(rep && --rep->ref_cnt==0)
	    delete rep;
	rep=copy.rep;
	if(rep)rep->ref_cnt++;
    }
    return *this;
}

template<class T>
Handle<T>& Handle<T>::operator=(T* crep)
{
    if(rep && --rep->ref_cnt==0)
	delete rep;
    rep=crep;
    if(rep)rep->ref_cnt++;
    return *this;
}

template<class T>
bool Handle<T>::operator==(const Handle<T>& crep) const
{
  return (get_rep() == crep.get_rep());
}

template<class T>
bool Handle<T>::operator!=(const Handle<T>& crep) const
{
  return (get_rep() != crep.get_rep());
}

template<class T>
Handle<T>::~Handle()
{
    if(rep && --rep->ref_cnt==0)
	delete rep;
}

template<class T>
void Handle<T>::detach()
{
    ASSERT(rep != 0);
    if(rep->ref_cnt==1){
	return; // We have the only copy
    }
    T* oldrep=rep;
    rep=oldrep->clone();
    oldrep->ref_cnt--;
    rep->ref_cnt++;
}

template<class T>
void Pio(Piostream& stream, Handle<T>& data)
{
    stream.begin_cheap_delim();
    Persistent* trep=data.get_rep();
    stream.io(trep, T::type_id);
    if(stream.reading()){
	data =(T*)trep;
    }
    stream.end_cheap_delim();
}

} // End namespace SCIRun


#endif


