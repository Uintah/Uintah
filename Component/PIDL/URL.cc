
/*
 *  URL.h: Abstraction for a URL
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Component/PIDL/URL.h>
#include <Component/PIDL/MalformedURL.h>
#include <sstream>

using Component::PIDL::URL;

URL::URL(const std::string& protocol,
		       const std::string& hostname,
		       int portno, const std::string& spec)
    : d_protocol(protocol), d_hostname(hostname),
      d_portno(portno), d_spec(spec)
{
}   

URL::URL(const std::string& str)
{
    // This is pretty simple minded, but it works for now
    int s=str.find("://");
    if(s == -1)
	throw MalformedURL(str, "No ://");
    d_protocol=str.substr(0, s);
    std::string rest=str.substr(s+3);
    s=rest.find(":");
    if(s == -1){
	s=rest.find("/");
	if(s==-1){
	    d_hostname=rest;
	    d_portno=0;
	    d_spec="";
	} else {
	    d_hostname=rest.substr(0, s);
	    d_spec=rest.substr(s+1);
	}
    } else {
	d_hostname=rest.substr(0, s);
	rest=rest.substr(s+1);
	std::istringstream i(rest);
	i >> d_portno;
	if(!i)
	    throw MalformedURL(str, "Error parsing port number");
	s=rest.find("/");
	if(s==-1){
	    d_spec="";
	} else {
	    d_spec=rest.substr(s+1);
	}
    }
}

URL::~URL()
{
}

std::string URL::getString() const
{
    std::ostringstream o;
    o << d_protocol << "://" << d_hostname;
    if(d_portno > 0)
	o << ":" << d_portno;
    if(d_spec.length() > 0 && d_spec[0] != '/')
	o << '/';
    o << d_spec;
    return o.str();
}

std::string URL::getProtocol() const
{
    return d_protocol;
}

std::string URL::getHostname() const
{
    return d_hostname;
}

int URL::getPortNumber() const
{
    return d_portno;
}

std::string URL::getSpec() const
{
    return d_spec;
}

//
// $Log$
// Revision 1.2  1999/08/31 08:59:02  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
// Revision 1.1  1999/08/30 17:39:49  sparker
// Updates to configure script:
//  rebuild configure if configure.in changes (Bug #35)
//  Fixed rule for rebuilding Makefile from Makefile.in (Bug #36)
//  Rerun configure if configure changes (Bug #37)
//  Don't build Makefiles for modules that aren't --enabled (Bug #49)
//  Updated Makfiles to build sidl and Component if --enable-parallel
// Updates to sidl code to compile on linux
// Imported PIDL code
// Created top-level Component directory
// Added ProcessManager class - a simpler interface to fork/exec (not finished)
//
//
