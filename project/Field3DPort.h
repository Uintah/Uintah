
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

class Field3D;
class Field3DHandle;

class Field3DIPort : public IPort {
public:
    enum Protocol {
	Atomic=0x01,
    };

protected:
    friend class Field3DOPort;
public:
    Field3DIPort(Module*, const clString& name, int protocol);
    virtual ~Field3DIPort();
    virtual void reset();
    virtual void finish();

    Field3DHandle get_field();
};

class Field3DOPort : public OPort {
    Field3DIPort* in;
public:
    Field3DOPort(Module*, const clString& name, int protocol);
    virtual ~Field3DOPort();

    virtual void reset();
    virtual void finish();

    void send_field(const Field3DHandle&);
};

#endif /* SCI_project_Field3DPort_h */
