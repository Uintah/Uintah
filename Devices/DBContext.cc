
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

#include <Devices/DBContext.h>

DBContext::DBContext(const clString& name)
: dialbox(0), name(name)
{
}

DBContext::~DBContext()
{
}

DBContext::Knob::Knob()
{
    min=value=0;
    max=1;
    scale=1;
    callback=0;
    rangetype=Bounded;
}

DBContext::Knob::~Knob()
{
}

void DBContext::set_knob(int which, const clString& name,
			 DBCallbackBase* callback)
{
    knobs[which].name=name;
    knobs[which].callback=callback;
}

void DBContext::set_range(int which, double min, double max)
{
    knobs[which].min=min;
    knobs[which].max=max;
    knobs[which].rangetype=Bounded;
}

void DBContext::set_scale(int which, double scale)
{
    knobs[which].rangetype=Unbounded;
    knobs[which].scale=scale;
}

void DBContext::set_wraprange(int which, double min, double max)
{
    knobs[which].min=min;
    knobs[which].max=max;
    knobs[which].rangetype=Wrapped;
}

void DBContext::set_value(int which, double value)
{
    knobs[which].value=value;
}

double DBContext::get_value(int which)
{
    return knobs[which].value;
}

DBCallbackBase::DBCallbackBase(Mailbox<MessageBase*>* mailbox, void* userdata)
: mailbox(mailbox), userdata(userdata)
{
}

DBCallbackBase::~DBCallbackBase()
{
}

void DBCallbackBase::dispatch(DBContext* context, int which, double value,
			      double delta)
{
    mailbox->send(new DBCallback_Message(this, context, which, value, delta, userdata));
}

DBCallback_Message::DBCallback_Message(DBCallbackBase* mcb, DBContext* context,
				       int which, double value, double delta,
				       void* cbdata)
: MessageBase(MessageTypes::DoDBCallback), mcb(mcb), context(context),
  which(which), value(value), delta(delta),  cbdata(cbdata)
{
}

DBCallback_Message::~DBCallback_Message()
{
}

