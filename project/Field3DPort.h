
/*
 *  Field3DPort.h: Handle to the Field3D Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Field3DPort_h
#define SCI_project_Field3DPort_h 1

#include <Port.h>
#include <Field3D.h>
#include <Multitask/ITC.h>

struct Field3DComm {
    Field3DComm();
    Field3DComm(const Field3DHandle&);
    Field3DHandle field;
    int has_field;
};

class Field3DIPort : public IPort {
    int recvd;
public:
    enum Protocol {
	Atomic=0x01,
    };

protected:
    friend class Field3DOPort;
    Mailbox<Field3DComm*> mailbox;
public:
    Field3DIPort(Module*, const clString& name, int protocol);
    virtual ~Field3DIPort();
    virtual void reset();
    virtual void finish();

    int get_field(Field3DHandle&);
};

class Field3DOPort : public OPort {
    Field3DIPort* in;
    int sent_something;
public:
    Field3DOPort(Module*, const clString& name, int protocol);
    virtual ~Field3DOPort();

    virtual void reset();
    virtual void finish();

    void send_field(const Field3DHandle&);
};

#endif /* SCI_project_Field3DPort_h */
