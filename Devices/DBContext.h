 
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
class DBCallbackBase;

class DBContext {
public:
    enum RangeType {
	Bounded,
	Unbounded,
	Wrapped,
    };
protected:
    friend class Dialbox;
    struct Knob {
	clString name;
	double min;
	double max;
	double scale;
	double value;
	RangeType rangetype;

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
    void set_wraprange(int, double, double);
    void set_value(int, double);
    void set_scale(int, double);
    double get_value(int);
};

class DBCallbackBase {
    Mailbox<MessageBase*>* mailbox;
    void* userdata;
public:
    DBCallbackBase(Mailbox<MessageBase*>* mailbox,
		   void* userdata);
    void dispatch(DBContext*, int, double, double);
    virtual ~DBCallbackBase();    
    virtual void perform(DBContext*, int, double, double, void*)=0;
};

class DBCallback_Message : public MessageBase {
public:
    DBCallbackBase* mcb;
    DBContext* context;
    int which;
    double value;
    double delta;
    void* cbdata;
    DBCallback_Message(DBCallbackBase* mcb, DBContext*, int, double, double, void* cbdata);
    virtual ~DBCallback_Message();
};

#endif
