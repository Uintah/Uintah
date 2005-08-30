/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  URL.h: Abstraction for a URL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include "URL.h"
#include <Core/CCA/PIDL/MalformedURL.h>
#include <iostream>
#include <sstream>
#include <netdb.h>
using namespace std;
using namespace SCIRun;

URL::URL(const string& protocol,
	 const string& hostname,
	 int portno, const string& spec)
    : d_protocol(protocol), d_hostname(hostname),
      d_portno(portno), d_spec(spec)
{
}   

URL::URL(const string& str)
{
    // This is pretty simple minded, but it works for now
    int s=str.find("://");
    if(s == -1)
	throw MalformedURL(str, "No ://");
    d_protocol=str.substr(0, s);
    string rest=str.substr(s+3);
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
	istringstream i(rest);
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

string URL::getString() const
{
    ostringstream o;
    o << d_protocol << "://" << d_hostname;
    if(d_portno > 0)
	o << ":" << d_portno;
    if(d_spec.length() > 0 && d_spec[0] != '/')
	o << '/';
    o << d_spec;
    return o.str();
}

string URL::getProtocol() const
{
    return d_protocol;
}

string URL::getHostname() const
{
    return d_hostname;
}

int URL::getPortNumber() const
{
    return d_portno;
}

string URL::getSpec() const
{
    return d_spec;
}

long URL::getIP(){
  struct hostent *he;
  if((he=gethostbyname(d_hostname.c_str())) == NULL){
    throw MalformedURL(d_hostname, "invalid hostname");
  }
  return *((long*)he->h_addr);
}

URL::URL(const URL& ucopy)
  : d_protocol(ucopy.d_protocol),
    d_hostname(ucopy.d_hostname),
    d_portno(ucopy.d_portno),
    d_spec(ucopy.d_spec)
{
}
