 
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

#include <tcl/tcl7.3/tcl.h>
#include <iostream.h>

extern Tcl_Interp* the_interp;

TCLvar::TCLvar(const clString& name, const clString& id,
	       TCL* tcl)
: varname(name+"("+id+")"), is_reset(1), tcl(tcl)
{
    tcl->register_var(this);
}

TCLvar::~TCLvar()
{
    tcl->unregister_var(this);
}

void TCLvar::reset()
{
    is_reset=1;
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

