
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

#ifndef SCI_Classlib_Handle_h
#define SCI_Classlib_Handle_h 1

#include <Classlib/Assert.h>

template<class T>
class Handle {
    T* rep;
public:
    inline Handle();
    inline Handle(T*);
    inline Handle(const Handle<T>&);
    inline Handle<T>& operator=(const Handle<T>&);
    inline Handle<T>& operator=(T*);
    inline ~Handle();

    inline T* operator->() const;
    inline T* get_rep() const;
};

template<class T>
inline Handle<T>::Handle()
: rep(0)
{
}

template<class T>
inline Handle<T>::Handle(T* rep)
: rep(rep)
{
    if(rep)rep->ref_cnt++;
}

template<class T>
inline Handle<T>::Handle(const Handle<T>& copy)
: rep(copy.rep)
{
    if(rep)rep->ref_cnt++;
}

template<class T>
inline Handle<T>& Handle<T>::operator=(const Handle<T>& copy)
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
inline Handle<T>& Handle<T>::operator=(T* crep)
{
    if(rep && --rep->ref_cnt==0)
	delete rep;
    rep=crep;
    if(rep)rep->ref_cnt++;
    return *this;
}

template<class T>
inline Handle<T>::~Handle()
{
    if(rep && --rep->ref_cnt==0)
	delete rep;
}

template<class T>
inline T* Handle<T>::operator->() const
{
    ASSERT(rep != 0);
    return rep;
}

template<class T>
inline T* Handle<T>::get_rep() const
{
    return rep;
}

#endif
