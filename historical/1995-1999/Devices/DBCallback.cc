
/*
 *  DBContext.h: Dialbox contexts
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifdef __GNUG__
#pragma interface
#endif

#include <Multitask/ITC.h>
#include <Devices/DBCallback.h>

#ifdef __GNUG__
#if 0
template<class T>
DBCallback<T>::DBCallback(Mailbox<MessageBase*>* mailbox, T* obj,
			  void (T::*method)(DBContext*, int, double, double, void*),
			  void* userdata)
: DBCallbackBase(mailbox, userdata), obj(obj), method(method)
{
}
#endif

#else

template<class T>
DBCallback<T>::DBCallback(Mailbox<MessageBase*>* mailbox, T* obj,
			  void (DBCallbackBase::*method)(DBContext*, int, double, double, void*),
			  void* userdata)
: DBCallbackBase(mailbox, userdata), obj(obj), method(method)
{
}
#endif
   
template<class T>
DBCallback<T>::~DBCallback()
{
}

template<class T>
void DBCallback<T>::perform(DBContext* ctx, int which,
			    double value, double delta, void* data)
{
#ifdef __GNUG__
    (obj->*method)(ctx, which, value, delta, data);
#else
    DBCallbackBase* dobj=(DBCallbackBase*)obj;
    (dobj->*method)(ctx, which, value, delta, data);
#endif
}

