/*
 *  GuiVar.cc: Interface to TCL variables
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Changes for distributed Dataflow:
 *   Michelle Miller 
 *   Thu May 14 01:24:12 MDT 1998
 * FIX: error cases and GuiVar* get()
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/GuiInterface/GuiVar.h>

#include <Core/GuiInterface/GuiManager.h>
#include <Core/GuiInterface/Remote.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <tcl.h>
#include <iostream>
using std::cerr;
using std::endl;
using std::ostream;

#ifdef _WIN32
#include <string.h>
#endif
#ifndef _WIN32
#include <values.h>
#else
#include <limits.h>
#include <float.h>
#endif

#ifndef MAXDOUBLE
#define MAXDOUBLE	DBL_MAX
#endif
#ifndef MAXINT
#define MAXINT		INT_MAX
#endif

#ifdef _WIN32
#undef ASSERT
#include <afxwin.h>
#define GLXContext HGLRC
#else
#include <GL/gl.h>
#include <GL/glx.h>
#endif

extern "C" Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

namespace SCIRun {

extern GuiManager* gm;

GuiVar::GuiVar(const clString& name, const clString& id,
	       TCL* tcl)
: varname(id+"-"+name), is_reset(1), tcl(tcl)
{
    if(tcl)
	tcl->register_var(this);
}

GuiVar::~GuiVar()
{
    if(tcl)
	tcl->unregister_var(this);
}

void GuiVar::reset()
{
    is_reset=1;
}

clString GuiVar::str()
{
    return varname;
}

clString GuiVar::format_varname()
{
  bool fixit=false;
  bool global=false;
  for(int i=0;i<varname.len();i++){
    if(!(varname.is_digit(i) || varname.is_alpha(i) || varname(i)=='_'
	 || varname(i) == '-' )){
      if( varname(i) == ':' ){
	global = true;
      } else {
	fixit=true;
	break;
      }
    }
  }
  if(fixit && global)
    return clString("{")+varname+"}";
  else if(fixit && !global)
    return clString("{::")+varname+"}";
  else if(!global) 
    return clString("::")+varname;
  else
    return varname;
}

GuiDouble::GuiDouble(const clString& name, const clString& id, TCL* tcl)
: GuiVar(name, id, tcl)
{
  value=-MAXDOUBLE;
}

GuiDouble::~GuiDouble()
{
}

double GuiDouble::get()
{
    // need access to network scope to get remote info.  I can't even look
    // up mod_id in hash table because the network has that.
    if(is_reset){
#ifndef _WIN32
	if (gm != NULL) {
            int skt = gm->getConnection();
#ifdef DEBUG
 	    cerr << "GuiDouble::get(): Got skt from gm->getConnection() = "
		 << skt << endl;
#endif
            // format request 
            TCLMessage msg;
	    msg.f = getDouble;
            strcpy (msg.tclName, varname());
            msg.un.tdouble = 0.0;

            // send request to server - no need for reply, error goes to Tk
            if (sendRequest (&msg, skt) == -1) {
                // error case ???
            }
            if (receiveReply (&msg, skt) == -1) {
		// error case ???
	    }
            gm->putConnection (skt);
	    value = msg.un.tdouble;
#ifdef DEBUG
	    cerr << "GuiDouble::get(): value from server = " << value << endl;
#endif

        } else {
#endif
	    TCLTask::lock();
	    char* l=Tcl_GetVar(the_interp, const_cast<char *>(varname()), TCL_GLOBAL_ONLY);
	    if(l){
	        Tcl_GetDouble(the_interp, l, &value);
	       	is_reset=0;
	    }
	    TCLTask::unlock();
#ifndef _WIN32
	}
#endif
    }
    return value;
}

void GuiDouble::set(double val)
{
    is_reset=0;
    if(val != value){
	TCLTask::lock();
	value=val;
	char buf[50];
	sprintf(buf, "%g", val);
	
	Tcl_SetVar(the_interp, const_cast<char *>(varname()), buf, TCL_GLOBAL_ONLY);
	TCLTask::unlock();
    }
}

void GuiDouble::emit(ostream& out)
{
    out << "set " << format_varname() << " " << get() << endl;
}

GuiInt::GuiInt(const clString& name, const clString& id, TCL* tcl)
: GuiVar(name, id, tcl)
{
  value=-MAXINT;
}

GuiInt::~GuiInt()
{
}

int GuiInt::get()
{
    if(is_reset){
#ifndef _WIN32
        if (gm != NULL) {
            int skt = gm->getConnection();
#ifdef DEBUG
	    cerr << "GuiInt::get(): Got skt from gm->getConnection() = "
		 << skt << endl;
#endif
            // format request
            TCLMessage msg;
            msg.f = getInt;
            strcpy (msg.tclName, varname());
            msg.un.tint = 0;

            // send request to server - no need for reply, error goes to Tk
            if (sendRequest (&msg, skt) == -1) {
                // error case ???
            }
            if (receiveReply (&msg, skt) == -1) {
                // error case ???
            }
            gm->putConnection (skt);
            value = msg.un.tint;
#ifdef DEBUG
	    cerr << "GuiInt::get(): value from server = " << value << endl;
#endif

        } else {
#endif

	    TCLTask::lock();
	    char* l=Tcl_GetVar(the_interp, const_cast<char *>(varname()), TCL_GLOBAL_ONLY);
	    if(l){
	        Tcl_GetInt(the_interp, l, &value);
	        is_reset=0;
	    }
	    TCLTask::unlock();
#ifndef _WIN32
	}
#endif
    }
    return value;
}

void GuiInt::set(int val)
{
    is_reset=0;
    if(val != value){
	TCLTask::lock();
	value=val;
	char buf[20];
	sprintf(buf, "%d", val);
	Tcl_SetVar(the_interp, const_cast<char *>(varname()), buf, TCL_GLOBAL_ONLY);
	TCLTask::unlock();
    }
}

void GuiInt::emit(ostream& out)
{
    out << "set " << format_varname() << " " << get() << endl;
}

GuiString::GuiString(const clString& name, const clString& id, TCL* tcl)
: GuiVar(name, id, tcl)
{
}

GuiString::~GuiString()
{
}

clString GuiString::get()
{
    if(is_reset){
#ifndef _WIN32
        if (gm != NULL) {
            int skt = gm->getConnection();
#ifdef DEBUG
	    cerr << "GuiString::get(): Got skt from gm->getConnection() = "
		 << skt << endl;
#endif

            // format request
            TCLMessage msg;
            msg.f = getString;
            strcpy (msg.tclName, varname());
            strcpy (msg.un.tstring, "");

            // send request to server - no need for reply, error goes to Tk
            if (sendRequest (&msg, skt) == -1) {
                // error case ???
            }
            if (receiveReply (&msg, skt) == -1) {
                // error case ???
            }
            gm->putConnection (skt);
	    value = clString(msg.un.tstring);
#ifdef DEBUG
	    cerr << "GuiString::get(): value from server = " << value << endl;
#endif

        } else {
#endif
	    TCLTask::lock();
	    char* l=Tcl_GetVar(the_interp, const_cast<char *>(varname()), TCL_GLOBAL_ONLY);
	    if(!l){
	        l="";
	    }
	    value=clString(l);
	    is_reset=0;
	    TCLTask::unlock();
#ifndef _WIN32
   	}
#endif
    }
    return value;
}

void GuiString::set(const clString& val)
{
    is_reset=0;
    if(val != value){
	TCLTask::lock();
	value=val;
	Tcl_SetVar(the_interp, const_cast<char *>(varname()), const_cast<char *>(value()), TCL_GLOBAL_ONLY);
	TCLTask::unlock();
    }
}

void GuiString::emit(ostream& out)
{
    out << "set " << format_varname() << " {" << get() << "}" << endl;
}

GuiVardouble::GuiVardouble(const clString& name, const clString& id, TCL* tcl)
: GuiVar(name, id, tcl)
{
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, const_cast<char *>(varname()), TCL_GLOBAL_ONLY);
    if(l){
	if(Tcl_GetDouble(the_interp, l, &value) != TCL_OK)
	    value=0;
    } else {
	value=0;
    }
    if(Tcl_LinkVar(the_interp, const_cast<char *>(varname()), (char*)&value, TCL_LINK_DOUBLE) != TCL_OK){
	cerr << "Error linking variable: " << varname << endl;
    }
    TCLTask::unlock();
}

GuiVardouble::~GuiVardouble()
{
    Tcl_UnlinkVar(the_interp, const_cast<char *>(varname()));
}

double GuiVardouble::get()
{
/*
    if (is_remote) {
    	// package remote request
        // send request over socket
	// block on reply
    } else {
 */
    	return value;
    //}
}

void GuiVardouble::emit(ostream& out)
{
    out << "set " << format_varname() << " " << get() << endl;
}

GuiVarint::GuiVarint(const clString& name, const clString& id, TCL* tcl)
: GuiVar(name, id, tcl)
{
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, const_cast<char *>(varname()), TCL_GLOBAL_ONLY);
    if(l){
	if(Tcl_GetInt(the_interp, l, &value) != TCL_OK)
	    value=0;
    } else {
	value=0;
    }
    if(Tcl_LinkVar(the_interp, const_cast<char *>(varname()), (char*)&value, TCL_LINK_INT) != TCL_OK){
	cerr << "Error linking variable: " << varname << endl;
    }
    TCLTask::unlock();
}

GuiVarint::~GuiVarint()
{
    Tcl_UnlinkVar(the_interp, const_cast<char *>(varname()));
}

int GuiVarint::get()
{
/*
    if (is_remote) {
        // package remote request
        // send request over socket
        // block on reply
    } else {
 */
        return value;
    // }
}

void GuiVarint::set(int nv)
{
    value=nv;
}

void GuiVardouble::set(double dv)
{
    value=dv;
}

void GuiVarint::emit(ostream& out)
{
    out << "set " << format_varname() << " " << get() << endl;
}

GuiVarintp::GuiVarintp(int* value,
		       const clString& name, const clString& id, TCL* tcl)
: GuiVar(name, id, tcl), value(value)
{
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, const_cast<char *>(varname()), TCL_GLOBAL_ONLY);
    if(l)
	Tcl_GetInt(the_interp, l, value);
    if(Tcl_LinkVar(the_interp, const_cast<char *>(varname()), (char*)value, TCL_LINK_INT) != TCL_OK){
	cerr << "Error linking variable: " << varname << endl;
    }
    TCLTask::unlock();
}

GuiVarintp::~GuiVarintp()
{
    Tcl_UnlinkVar(the_interp, const_cast<char *>(varname()));
}

// mm - must use Pio to get ptr
int GuiVarintp::get()
{
    return *value;
}

void GuiVarintp::set(int nv)
{
    *value=nv;
}

void GuiVarintp::emit(ostream& out)
{
    out << "set " << format_varname() << " " << get() << endl;
}

GuiPoint::GuiPoint(const clString& name, const clString& id, TCL* tcl)
: GuiVar(name, id, tcl), x("x", str(), tcl), y("y", str(), tcl),
  z("z", str(), tcl)
{
}

GuiPoint::~GuiPoint()
{
}

Point GuiPoint::get()
{
    return Point(x.get(), y.get(), z.get());
}

void GuiPoint::set(const Point& p)
{
    x.set(p.x());
    y.set(p.y());
    z.set(p.z());
}

void GuiPoint::emit(ostream& out)
{
    x.emit(out);
    y.emit(out);
    z.emit(out);
}

GuiVector::GuiVector(const clString& name, const clString& id, TCL* tcl)
: GuiVar(name, id, tcl), x("x", str(), tcl), y("y", str(), tcl),
  z("z", str(), tcl)
{
}

GuiVector::~GuiVector()
{
}

Vector GuiVector::get()
{
    return Vector(x.get(), y.get(), z.get());
}

void GuiVector::set(const Vector& p)
{
    x.set(p.x());
    y.set(p.y());
    z.set(p.z());
}

void GuiVector::emit(ostream& out)
{
    x.emit(out);
    y.emit(out);
    z.emit(out);
}

} // End namespace SCIRun
