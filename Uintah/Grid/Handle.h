
/*
 *  Handle.h: Smart Pointers..
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Containers_Handle_h
#define SCI_Containers_Handle_h

#include <SCICore/Util/Assert.h>

namespace Uintah {
    namespace Grid {
template<class T>
class Handle {
    T* rep;
public:
    Handle();
    Handle(T*);
    Handle(const Handle<T>&);
    Handle<T>& operator=(const Handle<T>&);
    Handle<T>& operator=(T*);
    ~Handle();

    void detach();

    inline const T* operator->() const {
	//ASSERT(rep != 0);
	return rep;
    }
    inline T* operator->() {
	//ASSERT(rep != 0);
	return rep;
    }
    inline T* get_rep() const {
	return rep;
    }
    inline operator bool() const {
	return rep != 0;
    }
};

template<class T>
Handle<T>::Handle()
: rep(0)
{
}

template<class T>
Handle<T>::Handle(T* rep)
: rep(rep)
{
    if(rep){
	rep->addReference();
    }
}

template<class T>
Handle<T>::Handle(const Handle<T>& copy)
: rep(copy.rep)
{
    if(rep){
	rep->addReference();
    }
}

template<class T>
Handle<T>& Handle<T>::operator=(const Handle<T>& copy)
{
    if(rep != copy.rep){
	if(rep){
	    if(rep->removeReference())
		delete rep;
	}
	rep=copy.rep;
	if(rep){
	    copy.rep->addReference();
	}
    }
    return *this;
}

template<class T>
Handle<T>& Handle<T>::operator=(T* copy)
{
    if(rep){
	if(rep->removeReference())
	    delete rep;
    }
    rep=copy;
    if(rep){
	rep->addReference();
    }
    return *this;
}

template<class T>
Handle<T>::~Handle()
{
    if(rep){
	if(rep->removeReference())
	    delete rep;
    }
}

template<class T>
void Handle<T>::detach()
{
    //ASSERT(rep != 0);
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

}
}

#endif
