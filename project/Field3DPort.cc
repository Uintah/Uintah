
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
#include <NotFinished.h>
#include <Port.h>
#include <Classlib/Assert.h>
#include <Classlib/String.h>
#include <iostream.h>

static clString Field3D_type("Field3D");
static clString Field3D_color("aquamarine4");

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

int Field3D::get_nx()
{
    return nx;
}

int Field3D::get_ny()
{
    return ny;
}

int Field3D::get_nz()
{
    return nz;
}

Field3D::Representation Field3D::get_rep()
{
    return rep;
}

Field3DHandle Field3DIPort::get_field()
{
    NOT_FINISHED("Field3DIPort::get_field");
    return Field3DHandle(0);
}

Field3DHandle::Field3DHandle(Field3D* rep)
: rep(rep)
{
    if(rep)
	rep->ref_cnt++;
}

Field3DHandle::~Field3DHandle()
{
    if(rep && --rep->ref_cnt==0)
	delete rep;
}

Field3D* Field3DHandle::operator->() const
{
    return rep;
}

