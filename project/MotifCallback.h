
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

#ifndef SCI_project_MotifCallback_h
#define SCI_project_MotifCallback_h 1

#include <MotifCallbackBase.h>
#include <Multitask/ITC.h>
class EncapsulatorC;

#define FIXCB(enc, cbname, mailbox, obj, meth, data, cloner) \
  (enc, cbname, mailbox, obj, (void (MotifCallbackBase::*)(CallbackData*, void*))meth, data, cloner)

template<class T> class MotifCallback : public MotifCallbackBase {
    T* obj;
    virtual void perform(CallbackData*, void*);
    void (MotifCallbackBase::*method)(CallbackData*, void*);
#if 0
    void (T::*method)(CallbackData*, void*);
#endif
public:
    MotifCallback(EncapsulatorC* enc, const char* cb_name,
		  Mailbox<MessageBase*>* mailbox, T* obj,
		  void (MotifCallbackBase::*method)(CallbackData*, void*),
		  void* userdata, CallbackData* (*cloner)(void*));
    MotifCallback(int delay, int repeat,
		  Mailbox<MessageBase*>* mailbox, T* obj,
		  void (MotifCallbackBase::*method)(CallbackData*, void*),
		  void* userdata, CallbackData* (*cloner)(void*));
#if 0
    MotifCallback(int delay, int repeat,
		  Mailbox<MessageBase*>* mailbox, T* obj,
		  void (T::*method)(CallbackData*, void*),
		  void* userdata=0);
    MotifCallback(EncapsulatorC* enc, const char* cb_name,
		  Mailbox<MessageBase*>* mailbox, T* obj,
		  void (T::*method)(CallbackData*, void*),
		  void* userdata=0);
#endif
    virtual ~MotifCallback();
};

#endif
