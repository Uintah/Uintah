/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

// DebugStream.cc - An ostream used for debug messages
// Written by:
// Eric Kuehne
// Department of Computer Science
// University of Utah
// Feb. 2000
// Copyright (C) 2000 SCI Group
// DebugStream is an ostream that is useful for outputing debug messages.
// When an instance is created, it is given a name.  An environment variable,
// SCI_DEBUG, is inspected to see if a particular instance should be
// active, and if so where to send the output.  The syntax for the
// environment variable is:
// SCI_DEBUG = ([name]:[-|+|+FILENAME])(,[name]:[-|+|+FILENAME])*
// The + or - specifies whether the named object is on or off.  If a file is 
// specified it is opened in ios::out mode.  If no file is specified,
// the stream is directed to cerr.  The : and , characters are
// restricted to deliminators.
// Example:
// SCI_DEBUG = modules.meshgen.warning:+meshgen.out,util.debugstream.error:-
// Future Additions:
// o Possible additions to constructor:
//   - Default file to output to
//   - Mode that the file will be opened in (append, out, etc.) (usefulness??)
// o Allow DEFAULT specification in the env variable which would override
//   all default settings. (usefulness??)
// o Time stamp option

#include <Core/Util/DebugStream.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>
using namespace std;

namespace SCIRun {

static const char *ENV_VAR = "SCI_DEBUG";


DebugBuf::DebugBuf()
{}


DebugBuf::~DebugBuf()
{}

int DebugBuf::overflow(int ch)
{
  if(owner->active()){
    return(*(owner->outstream) << (char)ch ? 0 : EOF);
  }
  return 0;
}


DebugStream::DebugStream(const string& iname, bool defaulton):
    std::ostream(0)
{
  dbgbuf = new DebugBuf();
  init(dbgbuf);
  name = iname;
  dbgbuf->owner = this;
  // set default values
  isactive = defaulton;
  if(isactive){
    outstream = &cerr;
  } else {
    outstream = 0;
  }
  // check SCI_DEBUG to see if this instance is mentioned
  checkenv(iname);
}


DebugStream::~DebugStream()
{
  if(outstream && outstream != &cerr){
    delete(outstream);
  }
  delete dbgbuf;
}

void DebugStream::checkenv(string iname)
{
  char* vars = getenv(ENV_VAR);
  if(!vars)
     return;
  string var(vars);
  // if SCI_DEBUG was defined, parse the string and store appropriate
  // values in onstreams and offstreams
  if(!var.empty()){
    string name, file;
    
    unsigned long oldcomma = 0;
    string::size_type commapos = var.find(',', 0);
    string::size_type colonpos = var.find(':', 0);
    if(commapos == string::npos){
      commapos = var.size();
    }
    while(colonpos != string::npos){
      name.assign(var, oldcomma, colonpos-oldcomma);
      if(name == iname){
	file.assign(var, colonpos+1, commapos-colonpos-1);
	if(file[0] == '-'){
	  isactive = false;
	  return;
	}
	else if(file[0] == '+'){
	  isactive = true;
	  // if no output file was specified, set to cerr
	  if(file.length() == 1){ 
	    outstream = &cerr;
	  }
	  else{
	    outstream = new ofstream(file.substr(1, file.size()-1).c_str());
	  }
	}
	else{
	  // houston, we have a problem: SCI_DEBUG was not correctly
	  // set.  Ignore and set all to default value	
	}
	return;
      }
      oldcomma = commapos+1;
      commapos = var.find(',', oldcomma+1);
      colonpos = var.find(':', oldcomma+1);
      if(commapos == string::npos){
	commapos = var.size();
      }
    }
  }
  else{
    // SCI_DEBUG was not defined,
    // all objects will be set to default
  }
}

} // End namespace SCIRun
