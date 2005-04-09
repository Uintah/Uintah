
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

#include <Classlib/LockingHandle.h>
#include <Classlib/Persistent.h>
#include <iostream.h>

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
    Persistent* trep;
    trep=data.rep;
    stream.io(trep, T::type_id);
    if(stream.reading()){
	data.rep=(T*)trep;
	if(data.rep)
	    data.rep->ref_cnt++;
    }
    stream.end_cheap_delim();
}
