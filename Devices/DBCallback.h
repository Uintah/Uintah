 
/*
 *  DBCallback.h: Dialbox callbacks
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef sci_project_DBCallback_h
#define sci_project_DBCallback_h 1

#ifdef __GNUG__
#pragma interface
#endif

#include <Devices/DBContext.h>

#ifdef __GNUG__
#define FIXCB2
#else
#define FIXCB2(mailbox, obj, meth, data) \
   (mailbox, obj, (void (DBCallbackBase::*)(DBContext*, int, double, double, void*))meth, data)
#endif

template<class T> class DBCallback : public DBCallbackBase {
    T* obj;
#ifdef __GNUG__
    void (T::*method)(DBContext*, int, double, double, void*);
#else
    void (DBCallbackBase::*method)(DBContext*, int, double, double, void*);
#endif
    virtual void perform(DBContext*, int, double, double, void*);
public:
#ifdef __GNUG__
    DBCallback(Mailbox<MessageBase*>* mailbox, T* obj,
	       void (T::*method)(DBContext*, int, double, double, void*),
	       void* userdata)
	: DBCallbackBase(mailbox, userdata), obj(obj), method(method)
	    {}
#else
    DBCallback(Mailbox<MessageBase*>* mailbox, T* obj,
	       void (DBCallbackBase::*method)(DBContext*, int, double, double, void*),
	       void*);
#endif
    virtual ~DBCallback();
};

#endif /* sci_project_DBCallback_h */
