
/*
 *  GeometryPort.cc: Handle to the Geometry Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <GeometryPort.h>
#include <Connection.h>
#include <NotFinished.h>
#include <Port.h>
#include <Classlib/Assert.h>
#include <Classlib/String.h>
#include <iostream.h>

static clString Geometry_type("Geometry");
static clString Geometry_color("magenta3");

GeometryIPort::GeometryIPort(Module* module, const clString& portname, int protocol)
: IPort(module, Geometry_type, portname, Geometry_color, protocol)
{
}

GeometryIPort::~GeometryIPort()
{
}

GeometryOPort::GeometryOPort(Module* module, const clString& portname, int protocol)
: OPort(module, Geometry_type, portname, Geometry_color, protocol)
{
}

GeometryOPort::~GeometryOPort()
{
}

void GeometryIPort::reset()
{
    NOT_FINISHED("GeometryIPort::reset");
}

void GeometryIPort::finish()
{
    NOT_FINISHED("GeometryIPort::finish");
}

void GeometryOPort::reset()
{
    NOT_FINISHED("GeometryOPort::reset");
}

void GeometryOPort::finish()
{
    NOT_FINISHED("GeometryOPort::finish");
}
