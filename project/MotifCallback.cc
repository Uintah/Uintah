
/*
 *  MotifCallback.h: Callbacks for a multi-threaded environment...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <MotifCallback.h>

template<class T>
MotifCallback<T>::MotifCallback(EncapsulatorC* enc, const char* cb_name,
				Mailbox<MessageBase*>* mailbox, T* obj,
				void (MotifCallbackBase::*method)(CallbackData*, void*),
				void* userdata, CallbackData* (*cloner)(void*))
: MotifCallbackBase(enc, cb_name, mailbox, userdata, cloner),
  obj(obj), method(method)
{
}

template<class T>
MotifCallback<T>::~MotifCallback()
{
}

template<class T>
void MotifCallback<T>::perform(CallbackData* cbdata, void* userdata)
{
    //(obj->*method)(d1, d2);
    MotifCallbackBase* mm=(MotifCallbackBase*)obj;
    (mm->*method)(cbdata, userdata);
}

