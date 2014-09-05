// DebugStream.cc - An ostream used for debug messages
//
// Written by:
// Eric Kuehne
// Department of Computer Science
// University of Utah
// Feb. 2000
//
// Copyright (C) 2000 SCI Group
//
// DebugStream is an ostream that is useful for outputing debug messages.
// When an instance is created, it is given a name.  An environment variable,
// SCI_DEBUG, is inspected to see if a particular instance should be
// active, and if so where to send the output.  The syntax for the
// environment variable is:
//
// SCI_DEBUG = ([name]:[-|+|+FILENAME])(,[name]:[-|+|+FILENAME])*
//
// The + or - specifies whether the named object is on or off.  If a file is 
// specified it is opened in ios::out mode.  If no file is specified,
// the stream is directed to cerr.  The : and , characters are
// restricted to deliminators.
//
// Example:
//
// SCI_DEBUG = modules.meshgen.warning:+meshgen.out,util.debugstream.error:-
//
// Future Additions:
// o Possible additions to constructor:
//   - Default file to output to
//   - Mode that the file will be opened in (append, out, etc.) (usefulness??)
// o Allow DEFAULT specification in the env variable which would override
//   all default settings. (usefulness??)
// o Time stamp option

#include "DebugStream.h"

namespace SCICore {
namespace Util {

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


DebugStream::DebugStream(string iname, bool defaulton):
  std::ostream(new DebugBuf()), std::ios(0)
{
  dbgbuf = new DebugBuf();
  init(dbgbuf);
  name = iname;
  dbgbuf->owner = this;
  // set default values
  isactive = defaulton;
  if(isactive){
    outstream = &cerr;
  }
  // check SCI_DEBUG to see if this instance is mentioned
  checkenv(iname);
}


DebugStream::~DebugStream()
{
  if(outstream != &cerr){
    delete(outstream);
  }
  delete dbgbuf;
}

void DebugStream::checkenv(string iname)
{
  char *env = getenv(ENV_VAR);

  // if SCI_DEBUG was defined, parse the string and store appropriate
  // values in onstreams and offstreams
  if( env ){
    string var = (string) env;
    string name, file; 
    int commapos, colonpos, oldcomma;
    commapos = colonpos = oldcomma = 0;
    commapos = var.find(',', 0);
    colonpos = var.find(':', 0);
    if(commapos == (int)string::npos){
      commapos = var.size();
    }
    while(colonpos != (int)string::npos){
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
      if(commapos == (int)string::npos){
	commapos = var.size();
      }
    }
  }
  else{
    // SCI_DEBUG was not defined,
    // all objects will be set to default
  }
}

} // end namespace SCICore
} // end namespace Util
