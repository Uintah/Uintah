
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

#ifndef sci_project_DBContext_h
#define sci_project_DBContext_h 1

#include <Multitask/ITC.h>
#include <Classlib/String.h>
#include <MessageBase.h>
class CallbackData;
class DBCallbackBase;

class DBContext {
protected:
    friend class Dialbox;
    struct Knob {
	clString name;
	clString value_string;
	double min;
	double max;
	double value;

	DBCallbackBase* callback;

	Knob();
	~Knob();
    };
    Knob knobs[8];

    Dialbox* dialbox;
    clString name;
public:
    DBContext(const clString& name);
    ~DBContext();

    void set_knob(int, const clString&,
		  DBCallbackBase*);
    void set_range(int, double, double);
    void set_value(int, double);
};

class DBCallbackBase {
    virtual void perform(DBContext*, int, void*)=0;
    Mailbox<MessageBase*>* mailbox;
    void* userdata;
public:
    DBCallbackBase(Mailbox<MessageBase*>* mailbox,
		   void* userdata);
    virtual ~DBCallbackBase();    
};

class DBCallback_Message : public MessageBase {
public:
    DBCallback_Message(DBCallbackBase* mcb, DBContext*, int);
    virtual ~DBCallback_Message();
    DBCallbackBase* mcb;
    CallbackData* cbdata;
};

#define FIXCB2(mailbox, obj, meth, data) \
   (mailbox, obj, (void (DBCallbackBase::*)(DBContext*, int, void*))meth, data)

template<class T> class DBCallback : public DBCallbackBase {
    T* obj;
    void (DBCallbackBase::*method)(DBContext*, int, void*);
#if 0
    void (T::*method)(DBContext*, int, void*);
#endif
    virtual void perform(DBContext*, int, void*);
public:
    DBCallback(Mailbox<MessageBase*>* mailbox, T* obj,
	       void (DBCallbackBase::*method)(DBContext*, int, void*),
	       void*);
#if 0
    DBCallback(Mailbox<MessageBase*>* mailbox, T* obj,
	       void (T::*method)(DBContext*, int, void*),
	       void*);
#endif
    virtual ~DBCallback();
};

#endif

