 
/*
 *  TCLvar.cc: Interface to TCL variables
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <TCL/TCL.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>
#include <Geom/Color.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

#include <tcl/tcl/tcl.h>
#include <iostream.h>

extern Tcl_Interp* the_interp;

TCLvar::TCLvar(const clString& name, const clString& id,
	       TCL* tcl)
: varname(name+","+id), is_reset(1), tcl(tcl)
{
    if(tcl)
	tcl->register_var(this);
}

TCLvar::~TCLvar()
{
    if(tcl)
	tcl->unregister_var(this);
}

void TCLvar::reset()
{
    is_reset=1;
}

clString TCLvar::str()
{
    return varname;
}

TCLdouble::TCLdouble(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl)
{
}

TCLdouble::~TCLdouble()
{
}

double TCLdouble::get()
{
    if(is_reset){
	TCLTask::lock();
	char* l=Tcl_GetVar(the_interp, varname(), TCL_GLOBAL_ONLY);
	if(l){
	    Tcl_GetDouble(the_interp, l, &value);
	    is_reset=0;
	}
	TCLTask::unlock();
    }
    return value;
}

void TCLdouble::set(double val)
{
    is_reset=0;
    if(val != value){
	TCLTask::lock();
	value=val;
	char buf[20];
	sprintf(buf, "%f", val);
	Tcl_SetVar(the_interp, varname(), buf, TCL_GLOBAL_ONLY);
	TCLTask::unlock();
    }
}

TCLint::TCLint(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl)
{
}

TCLint::~TCLint()
{
}

int TCLint::get()
{
    if(is_reset){
	TCLTask::lock();
	char* l=Tcl_GetVar(the_interp, varname(), TCL_GLOBAL_ONLY);
	if(l){
	    Tcl_GetInt(the_interp, l, &value);
	    is_reset=0;
	}
	TCLTask::unlock();
    }
    return value;
}

void TCLint::set(int val)
{
    is_reset=0;
    if(val != value){
	TCLTask::lock();
	value=val;
	char buf[20];
	sprintf(buf, "%d", val);
	Tcl_SetVar(the_interp, varname(), buf, TCL_GLOBAL_ONLY);
	TCLTask::unlock();
    }
}

TCLstring::TCLstring(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl)
{
}

TCLstring::~TCLstring()
{
}

clString TCLstring::get()
{
    if(is_reset){
	TCLTask::lock();
	char* l=Tcl_GetVar(the_interp, varname(), TCL_GLOBAL_ONLY);
	if(!l){
	    l="";
	}
	value=clString(l);
	is_reset=0;
	TCLTask::unlock();
    }
    return value;
}

void TCLstring::set(const clString& val)
{
    is_reset=0;
    if(val != value){
	TCLTask::lock();
	value=val;
	Tcl_SetVar(the_interp, varname(), value(), TCL_GLOBAL_ONLY);
	TCLTask::unlock();
    }
}


TCLvardouble::TCLvardouble(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl)
{
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, varname(), TCL_GLOBAL_ONLY);
    if(l){
	if(Tcl_GetDouble(the_interp, l, &value) != TCL_OK)
	    value=0;
    } else {
	value=0;
    }
    if(Tcl_LinkVar(the_interp, varname(), (char*)&value, TCL_LINK_DOUBLE) != TCL_OK){
	cerr << "Error linking variable: " << varname << endl;
    }
    TCLTask::unlock();
}

TCLvardouble::~TCLvardouble()
{
    Tcl_UnlinkVar(the_interp, varname());
}

double TCLvardouble::get()
{
    return value;
}

TCLvarint::TCLvarint(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl)
{
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, varname(), TCL_GLOBAL_ONLY);
    if(l){
	if(Tcl_GetInt(the_interp, l, &value) != TCL_OK)
	    value=0;
    } else {
	value=0;
    }
    if(Tcl_LinkVar(the_interp, varname(), (char*)&value, TCL_LINK_INT) != TCL_OK){
	cerr << "Error linking variable: " << varname << endl;
    }
    TCLTask::unlock();
}

TCLvarint::~TCLvarint()
{
    Tcl_UnlinkVar(the_interp, varname());
}

int TCLvarint::get()
{
    return value;
}

void TCLvarint::set(int nv)
{
    value=nv;
}

TCLvarintp::TCLvarintp(int* value,
		       const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl), value(value)
{
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, varname(), TCL_GLOBAL_ONLY);
    if(Tcl_LinkVar(the_interp, varname(), (char*)value, TCL_LINK_INT) != TCL_OK){
	cerr << "Error linking variable: " << varname << endl;
    }
    TCLTask::unlock();
}

TCLvarintp::~TCLvarintp()
{
    Tcl_UnlinkVar(the_interp, varname());
}

int TCLvarintp::get()
{
    return *value;
}

void TCLvarintp::set(int nv)
{
    *value=nv;
}

TCLPoint::TCLPoint(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl), x("x", str(), tcl), y("y", str(), tcl),
  z("z", str(), tcl)
{
}

TCLPoint::~TCLPoint()
{
}

Point TCLPoint::get()
{
    return Point(x.get(), y.get(), z.get());
}

void TCLPoint::set(const Point& p)
{
    x.set(p.x());
    y.set(p.y());
    z.set(p.z());
}

TCLVector::TCLVector(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl), x("x", str(), tcl), y("y", str(), tcl),
  z("z", str(), tcl)
{
}

TCLVector::~TCLVector()
{
}

Vector TCLVector::get()
{
    return Vector(x.get(), y.get(), z.get());
}

void TCLVector::set(const Vector& p)
{
    x.set(p.x());
    y.set(p.y());
    z.set(p.z());
}

TCLColor::TCLColor(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl), r("r", str(), tcl), g("g", str(), tcl),
  b("b", str(), tcl)
{
}

TCLColor::~TCLColor()
{
}

Color TCLColor::get()
{
    return Color(r.get(), g.get(), b.get());
}

void TCLColor::set(const Color& p)
{
    r.set(p.r());
    g.set(p.g());
    b.set(p.b());
}

