
/*
 *  Dialbox.h: Dialbox manager thread...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef sci_project_Dialbox_h
#define sci_project_Dialbox_h 1

#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <DBContext.h>

struct DialMsg : public MessageBase {
    DBContext* context;
    DialMsg(DBContext*);
    int which;
    int info;
    DialMsg(int, int);
    ~DialMsg();
};

class Dialbox : public Task {
    DBContext* context;
    Mailbox<MessageBase*> mailbox;

public:
    Dialbox();
    virtual ~Dialbox();

    virtual int body(int);

    static void attach_dials(DBContext*);
    static int get_event_type();
    static void handle_event(void*);
};

#endif
