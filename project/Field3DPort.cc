
/*
 *  Field3DPort.cc: Handle to the Field3D Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Field3DPort.h>
#include <Connection.h>
#include <Field3D.h>
#include <NotFinished.h>
#include <Port.h>
#include <Classlib/Assert.h>
#include <Classlib/String.h>
#include <iostream.h>

static clString Field3D_type("Field3D");
static clString Field3D_color("VioletRed2");

Field3DIPort::Field3DIPort(Module* module, const clString& portname, int protocol)
: IPort(module, Field3D_type, portname, Field3D_color, protocol)
{
}

Field3DIPort::~Field3DIPort()
{
}

Field3DOPort::Field3DOPort(Module* module, const clString& portname, int protocol)
: OPort(module, Field3D_type, portname, Field3D_color, protocol)
{
}

Field3DOPort::~Field3DOPort()
{
}

void Field3DIPort::reset()
{
    NOT_FINISHED("Field3DIPort::reset()");
}

void Field3DIPort::finish()
{
    NOT_FINISHED("Field3DIPort::finish()");
}

void Field3DOPort::reset()
{
    NOT_FINISHED("Field3DOPort::reset()");
}

void Field3DOPort::finish()
{
    NOT_FINISHED("Field3DOPort::reset()");
}

void Field3DOPort::send_field(const Field3DHandle& field)
{
    NOT_FINISHED("send_field");
}

Field3DHandle Field3DIPort::get_field()
{
    NOT_FINISHED("Field3DIPort::get_field");
    return Field3DHandle(0);
}
