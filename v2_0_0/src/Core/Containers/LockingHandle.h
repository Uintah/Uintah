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
 *  LockingHandle.h: Smart Pointers..
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Containers_LockingHandle_h
#define SCI_Containers_LockingHandle_h 1

#include <sci_config.h>

#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>

namespace SCIRun {


template<class T>
class LockingHandle;
template<class T>
void Pio(Piostream& stream, LockingHandle<T>& data);

template<class T>
class LockingHandle {
  T* rep;
public:

  typedef T   value_type;
  typedef T * pointer_type;

  LockingHandle();
  LockingHandle(T*);
  LockingHandle(const LockingHandle<T>&);
  LockingHandle<T>& operator=(const LockingHandle<T>&);
  LockingHandle<T>& operator=(T*);
  bool operator==(const LockingHandle<T>&) const;
  bool operator!=(const LockingHandle<T>&) const;

  ~LockingHandle();

  void detach();

  inline T* operator->() const { ASSERT(rep != 0); return rep; }
  inline T* get_rep() const { return rep; }


#if defined(_AIX)
  template <typename Type> 
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream& stream,LockingHandle<Type>& data);
#else
  friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream& stream,LockingHandle<T>& data);
#endif
};

template<class T>
LockingHandle<T>::LockingHandle()
: rep(0)
{
}

template<class T>
LockingHandle<T>::LockingHandle(T* rep)
: rep(rep)
{
    if(rep){
	rep->lock.lock();
	rep->ref_cnt++;
	rep->lock.unlock();
    }
}

template<class T>
LockingHandle<T>::LockingHandle(const LockingHandle<T>& copy)
: rep(copy.rep)
{
    if(rep){
	rep->lock.lock();
	rep->ref_cnt++;
	rep->lock.unlock();
    }
}

template<class T>
LockingHandle<T>& LockingHandle<T>::operator=(const LockingHandle<T>& copy)
{
    if(rep != copy.rep){
	if(rep){
	    rep->lock.lock();
	    if(--rep->ref_cnt==0){
		rep->lock.unlock();
		delete rep;
	    } else {
		rep->lock.unlock();
	    }
	}
	if(copy.rep){
	    copy.rep->lock.lock();
	    rep=copy.rep;
	    rep->ref_cnt++;
	    copy.rep->lock.unlock();
	} else {
	    rep=copy.rep;
	}
    }
    return *this;
}

template<class T>
LockingHandle<T>& LockingHandle<T>::operator=(T* crep)
{
    if(rep){
	rep->lock.lock();
	if(--rep->ref_cnt==0){
	    rep->lock.unlock();
	    delete rep;
	} else {
	    rep->lock.unlock();
	}
    }
    if(crep){
	crep->lock.lock();
	rep=crep;
	rep->ref_cnt++;
	crep->lock.unlock();
    } else {
	rep=crep;
    }
    return *this;
}

template<class T>
bool LockingHandle<T>::operator==(const LockingHandle<T>& crep) const
{
  return (get_rep() == crep.get_rep());
}

template<class T>
bool LockingHandle<T>::operator!=(const LockingHandle<T>& crep) const
{
  return (get_rep() != crep.get_rep());
}

template<class T>
LockingHandle<T>::~LockingHandle()
{
    if(rep){
	rep->lock.lock();
	if(--rep->ref_cnt==0){
	    rep->lock.unlock();
	    delete rep;
	} else {
	    rep->lock.unlock();
	}
    }
}

template<class T>
void LockingHandle<T>::detach()
{
    ASSERT(rep != 0);
    rep->lock.lock();
    if(rep->ref_cnt==1){
	rep->lock.unlock();
	return; // We have the only copy
    }
    T* oldrep=rep;
    rep=oldrep->clone();
    oldrep->ref_cnt--;
    oldrep->lock.unlock();
    rep->ref_cnt++;
}

template<class T>
void Pio(Piostream& stream, LockingHandle<T>& data)
{
    stream.begin_cheap_delim();
    Persistent* trep=data.rep;
    stream.io(trep, T::type_id);
    if(stream.reading()){
	data.rep=(T*)trep;
	if(data.rep)
	    data.rep->ref_cnt++;
    }
    stream.end_cheap_delim();
}

} // End namespace SCIRun


#endif
