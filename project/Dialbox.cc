
/*
 *  Dialbox.cc: Dialbox manager thread...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dialbox.h>
#include <NotFinished.h>
#include <iostream.h>

static Dialbox* the_dialbox;

Dialbox::Dialbox(ColorManager* color_manager)
: Task("Dialbox", 1), context(0), color_manager(color_manager),
  mailbox(10)
{
}

Dialbox::~Dialbox()
{
}

int Dialbox::body(int)
{
    if(the_dialbox){
	cerr << "There should be only one Dialbox thread!\n";
	return 0;
    }
    the_dialbox=this;
    while(1){
	mailbox.receive();
	NOT_FINISHED("Dialbox::body()");
    }
    return 0;
}

void Dialbox::attach_dials(DBContext* context)
{
    if(!the_dialbox){
	cerr << "The dialbox hasn't been created yet!!!" << endl;
	return;
    }
    the_dialbox->mailbox.send(new DialMsg(context));
}

DialMsg::DialMsg(DBContext* context)
: context(context), what(Attach)
{
}
