
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

#include <MotifCallbackBase.h>
#include <Mt/Encapsulator.h>
#include <MtXEventLoop.h>
#include <stdio.h>
extern MtXEventLoop* evl;

static int motif_translation_installed=0;
DECLARECALLBACKC(MotifCallbackBase, MotifCallbackH);
DEFINECALLBACKC(MotifCallbackBase, MotifCallbackH);

class MotifCallbackHelper : public MotifCallbackH {
    EncapsulatorC* enc;
public:
    MotifCallbackHelper(EncapsulatorC* enc, const char* cb_name,
			MotifCallbackBase* ptr,
			void (MotifCallbackBase::*pmf)(void*))
	: MotifCallbackH(enc, cb_name, ptr, pmf), enc(enc){}
    void maybe_install();
};

void MotifCallbackHelper::maybe_install()
{
    if(*enc){
	widget_=*enc;
	Install();
    }
}

static void handle_translation(Widget, XEvent* event, String* params,
			       Cardinal* num_params)
{
    if(*num_params != 1){
	cerr << "Handle translation is messed up...\n";
	return;
    }
    MotifCallbackBase* cb;
    sscanf(params[0], "%p", &cb);
    cb->dispatch(event);
}

MotifCallbackBase::MotifCallbackBase(EncapsulatorC* enc, const char* cb_name,
				     Mailbox<MessageBase*>* mailbox,
				     void* userdata,
				     CallbackData* (*cloner)(void*))
: enc(enc), mailbox(mailbox), userdata(userdata), cloner(cloner)
{
    // Install the callback on the Encapsulator...
    if(cb_name[0] == '<'){
	// A translation...
	static char* tname="general_motif_translation";
	evl->lock();
	if(!motif_translation_installed){
	    XtActionsRec actions;
	    actions.string=tname;
	    actions.proc=handle_translation;
	    XtAppAddActions(evl->get_app(), &actions, 1);
	    motif_translation_installed=1;
	}
	char transl[1000];
	sprintf(transl, "%s: %s(%p)", cb_name, tname, this);
	XtOverrideTranslations(*enc, XtParseTranslationTable(transl));
	evl->unlock();
    } else {
	MotifCallbackHelper* cb=new MotifCallbackHelper(enc, cb_name, this,
						    &MotifCallbackBase::dispatch);
	cb->maybe_install();
    }
}

void MotifCallbackBase::handle_timeout(void* ud, void* data)
{
    MotifCallbackBase* that=(MotifCallbackBase*)ud;
    if(that->repeat)
       XtAppAddTimeOut(evl->get_app(), that->repeat,
		       (XtTimerCallbackProc)handle_timeout, that);
    CallbackData* cbdata=0;
    if(that->cloner)
	cbdata=(*that->cloner)(data);
    that->mailbox->send(new Callback_Message(that, cbdata));
}

MotifCallbackBase::MotifCallbackBase(int delay, int repeat,
				     Mailbox<MessageBase*>* mailbox,
				     void* userdata,
				     CallbackData* (*cloner)(void*))
: delay(delay), repeat(repeat),
  mailbox(mailbox), userdata(userdata), cloner(cloner)
{
    if(!delay)delay=repeat;
    XtAppAddTimeOut(evl->get_app(), delay,
		    (XtTimerCallbackProc)handle_timeout, this);
}

MotifCallbackBase::~MotifCallbackBase()
{
}

void MotifCallbackBase::perform(CallbackData* cbdata)
{
    perform(cbdata, userdata);
}

void MotifCallbackBase::dispatch(void* data)
{
    // We need to clone the data before we send it along.
    // The callback provides the mechanism of how to do that
    CallbackData* cbdata=0;
    if(cloner)
	cbdata=(*cloner)(data);
    mailbox->send(new Callback_Message(this, cbdata));
}

Callback_Message::Callback_Message(MotifCallbackBase* mcb, CallbackData* cbdata)
: MessageBase(MessageTypes::DoCallback), mcb(mcb), cbdata(cbdata)
{
}

Callback_Message::~Callback_Message()
{
}

