//static char *id="@(#) $Id$";
 
/*
 *  TCLvar.cc: Interface to TCL variables
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Changes for distributed SCIRun:
 *   Michelle Miller 
 *   Thu May 14 01:24:12 MDT 1998
 * FIX: error cases and TCLvar* get()
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <TclInterface/TCLvar.h>

#include <TclInterface/GuiManager.h>
#include <TclInterface/Remote.h>
#include <TclInterface/TCL.h>
#include <TclInterface/TCLTask.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

#include <tcl.h>
#include <iostream.h>
#include <values.h>

extern Tcl_Interp* the_interp;

namespace SCICore {
namespace TclInterface {

extern GuiManager* gm;

TCLvar::TCLvar(const clString& name, const clString& id,
	       TCL* tcl)
: varname(id+"-"+name), is_reset(1), tcl(tcl)
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

clString TCLvar::format_varname()
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

TCLdouble::TCLdouble(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl)
{
  value=-MAXDOUBLE;
}

TCLdouble::~TCLdouble()
{
}

double TCLdouble::get()
{
    // need access to network scope to get remote info.  I can't even look
    // up mod_id in hash table because the network has that.
    if(is_reset){
	if (gm != NULL) {
            int skt = gm->getConnection();
#ifdef DEBUG
 	    cerr << "TCLdouble::get(): Got skt from gm->getConnection() = "
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
	    cerr << "TCLdouble::get(): value from server = " << value << endl;
#endif

        } else {
	    TCLTask::lock();
	    char* l=Tcl_GetVar(the_interp, const_cast<char *>(varname()), TCL_GLOBAL_ONLY);
	    if(l){
	        Tcl_GetDouble(the_interp, l, &value);
	       	is_reset=0;
	    }
	    TCLTask::unlock();
	}
    }
    return value;
}

void TCLdouble::set(double val)
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

void TCLdouble::emit(ostream& out)
{
    out << "set " << format_varname() << " " << get() << endl;
}

TCLint::TCLint(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl)
{
  value=-MAXINT;
}

TCLint::~TCLint()
{
}

int TCLint::get()
{
    if(is_reset){
        if (gm != NULL) {
            int skt = gm->getConnection();
#ifdef DEBUG
	    cerr << "TCLint::get(): Got skt from gm->getConnection() = "
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
	    cerr << "TCLint::get(): value from server = " << value << endl;
#endif

        } else {

	    TCLTask::lock();
	    char* l=Tcl_GetVar(the_interp, const_cast<char *>(varname()), TCL_GLOBAL_ONLY);
	    if(l){
	        Tcl_GetInt(the_interp, l, &value);
	        is_reset=0;
	    }
	    TCLTask::unlock();
	}
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
	Tcl_SetVar(the_interp, const_cast<char *>(varname()), buf, TCL_GLOBAL_ONLY);
	TCLTask::unlock();
    }
}

void TCLint::emit(ostream& out)
{
    out << "set " << format_varname() << " " << get() << endl;
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
        if (gm != NULL) {
            int skt = gm->getConnection();
#ifdef DEBUG
	    cerr << "TCLstring::get(): Got skt from gm->getConnection() = "
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
	    cerr << "TCLstring::get(): value from server = " << value << endl;
#endif

        } else {
	    TCLTask::lock();
	    char* l=Tcl_GetVar(the_interp, const_cast<char *>(varname()), TCL_GLOBAL_ONLY);
	    if(!l){
	        l="";
	    }
	    value=clString(l);
	    is_reset=0;
	    TCLTask::unlock();
   	}
    }
    return value;
}

void TCLstring::set(const clString& val)
{
    is_reset=0;
    if(val != value){
	TCLTask::lock();
	value=val;
	Tcl_SetVar(the_interp, const_cast<char *>(varname()), const_cast<char *>(value()), TCL_GLOBAL_ONLY);
	TCLTask::unlock();
    }
}

void TCLstring::emit(ostream& out)
{
    out << "set " << format_varname() << " {" << get() << "}" << endl;
}

TCLvardouble::TCLvardouble(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl)
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

TCLvardouble::~TCLvardouble()
{
    Tcl_UnlinkVar(the_interp, const_cast<char *>(varname()));
}

double TCLvardouble::get()
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

void TCLvardouble::emit(ostream& out)
{
    out << "set " << format_varname() << " " << get() << endl;
}

TCLvarint::TCLvarint(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl)
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

TCLvarint::~TCLvarint()
{
    Tcl_UnlinkVar(the_interp, const_cast<char *>(varname()));
}

int TCLvarint::get()
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

void TCLvarint::set(int nv)
{
    value=nv;
}

void TCLvarint::emit(ostream& out)
{
    out << "set " << format_varname() << " " << get() << endl;
}

TCLvarintp::TCLvarintp(int* value,
		       const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl), value(value)
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

TCLvarintp::~TCLvarintp()
{
    Tcl_UnlinkVar(the_interp, const_cast<char *>(varname()));
}

// mm - must use Pio to get ptr
int TCLvarintp::get()
{
    return *value;
}

void TCLvarintp::set(int nv)
{
    *value=nv;
}

void TCLvarintp::emit(ostream& out)
{
    out << "set " << format_varname() << " " << get() << endl;
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

void TCLPoint::emit(ostream& out)
{
    x.emit(out);
    y.emit(out);
    z.emit(out);
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

void TCLVector::emit(ostream& out)
{
    x.emit(out);
    y.emit(out);
    z.emit(out);
}

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:16  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:11:04  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/26 19:21:41  kuzimmer
// Added global namespace ids (::) to the format_varname routine -Kurt
//
// Revision 1.2  1999/05/17 17:14:47  kuzimmer
// Added the format_variable function from SCIRun
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//
