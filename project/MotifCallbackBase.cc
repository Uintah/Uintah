
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

DECLARECALLBACKC(MotifCallbackBase, MotifCallbackHelper);
DEFINECALLBACKC(MotifCallbackBase, MotifCallbackHelper);

MotifCallbackBase::MotifCallbackBase(EncapsulatorC* enc, const char* cb_name,
				     Mailbox<MessageBase*>* mailbox,
				     void* userdata,
				     CallbackData* (*cloner)(void*))
: enc(enc), mailbox(mailbox), userdata(userdata), cloner(cloner)
{
    // Install the callback on the Encapsulator...
    new MotifCallbackHelper(enc, cb_name, this, &MotifCallbackBase::dispatch);
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

