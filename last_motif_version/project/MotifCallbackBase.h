
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

#ifndef SCI_project_MotifCallbackBase_h
#define SCI_project_MotifCallbackBase_h 1

#include <Multitask/ITC.h>
#include <MessageBase.h>

class CallbackData;
class EncapsulatorC;

class MotifCallbackBase {
    int delay, repeat;
    EncapsulatorC* enc;
    Mailbox<MessageBase*>* mailbox;
    void* userdata;
    virtual void perform(CallbackData*, void*)=0;
    CallbackData* (*cloner)(void*);
    static void handle_timeout(void*, void*);
public:
    MotifCallbackBase(int delay, int repeat,
		      Mailbox<MessageBase*>* mailbox,
		      void* userdata, CallbackData* (*cloner)(void*));
    MotifCallbackBase(EncapsulatorC* enc, const char* cb_name,
		      Mailbox<MessageBase*>* mailbox,
		      void* userdata, CallbackData* (*cloner)(void*));
    virtual ~MotifCallbackBase();
    void perform(CallbackData*);
    void dispatch(void*);
};

class Callback_Message : public MessageBase {
public:
    Callback_Message(MotifCallbackBase* mcb, CallbackData* cbdata);
    virtual ~Callback_Message();
    MotifCallbackBase* mcb;
    CallbackData* cbdata;
};

#endif
