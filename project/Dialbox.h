
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
class ColorManager;

struct DialMsg {
    enum What {
	Attach,
    };
    What what;
    DBContext* context;
    DialMsg(DBContext*);
};

class Dialbox : public Task {
    ColorManager* color_manager;
    DBContext* context;
    Mailbox<DialMsg*> mailbox;
public:
    Dialbox(ColorManager*);
    virtual ~Dialbox();

    virtual int body(int);

    static void attach_dials(DBContext*);
};

#endif
