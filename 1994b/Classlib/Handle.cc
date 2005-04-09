
/*
 *  Handle.cc: Smart Pointers..
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/Handle.h>

#ifdef __GNUG__
#pragma interface
#endif

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
Handle<T>::~Handle()
{
    if(rep && --rep->ref_cnt==0)
	delete rep;
}

template<class T>
T* Handle<T>::operator->() const
{
    ASSERT(rep != 0);
    return rep;
}

template<class T>
T* Handle<T>::get_rep() const
{
    return rep;
}

