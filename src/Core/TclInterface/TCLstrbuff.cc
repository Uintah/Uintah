/*
 *  TCLstrbuff.cc: implementation of string buffer for TclStream itcl-class 
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   January, 2000
 *   
 *   Copyright (C) 2000 SCI Group
 */

#include <Core/TclInterface/TCLstrbuff.h>
#include <Core/TclInterface/TCLTask.h>
#include <string.h>
#include <tcl.h>

extern "C" Tcl_Interp* the_interp;

namespace SCIRun {

using namespace std;

////////////
// TCLstrbuff implementation

//////////
// Constructor/Destructor
TCLstrbuff::TCLstrbuff(const clString& name, const clString& id, TCL* tcl):
  TCLvar(name, id, tcl),
  ostringstream()
{
  TCLTask::lock();
  
  d_buff = Tcl_Alloc(d_bSize=256);
  *d_buff='\0';
  
  if ( Tcl_LinkVar(the_interp, const_cast<char *>(varname()), (char*)&d_buff, TCL_LINK_STRING | TCL_LINK_READ_ONLY)!=TCL_OK){
    cout << "Not possible to link tcl var" << std::endl;
  }
  
  TCLTask::unlock();
}

TCLstrbuff::~TCLstrbuff(){
  Tcl_UnlinkVar(the_interp, const_cast<char *>(varname()));
  Tcl_UnsetVar(the_interp, const_cast<char*>(varname()), TCL_GLOBAL_ONLY);
  Tcl_Free(d_buff);
}

//////////
//   
void TCLstrbuff::emit(std::ostream&){
  
}

//////////
//
TCLstrbuff& TCLstrbuff::flush(){
 
  TCLTask::lock();

  const int n = tellp();
  if (n>d_bSize)
    d_buff = Tcl_Realloc(d_buff, d_bSize = n);
  
  strcpy(d_buff, ostringstream::str().c_str()); 
  Tcl_UpdateLinkedVar(the_interp, const_cast<char*>(varname())); 
  TCLTask::unlock();
  
  // reinitializing the stream
  ostringstream::clear();
  ostringstream::seekp(0);
  ostringstream::str("");
  return *this;
}

TCLstrbuff& operator<<(TCLstrbuff& stream, TCLstrbuff& (*mp)(TCLstrbuff&)){
  return mp(stream);
}

TCLstrbuff& endl(TCLstrbuff& stream){
  static_cast<ostringstream&>(stream)<<'\n';
  stream.flush();
  return stream;
}

TCLstrbuff& flush(TCLstrbuff& stream){
  stream.flush();
  return stream;
}

TCLstrbuff& ends(TCLstrbuff& stream){
  static_cast<ostringstream&>(stream)<<'\0';
  stream.flush();
  return stream;
}

} // end namespace SCIRun
