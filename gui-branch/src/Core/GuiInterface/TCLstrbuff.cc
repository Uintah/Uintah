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

#include <Core/GuiInterface/TCLstrbuff.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Containers/StringUtil.h>
#include <string.h>
#include <tcl.h>

extern "C" Tcl_Interp* the_interp;

namespace SCIRun {

using namespace std;

////////////
// TCLstrbuff implementation

//////////
// Constructor/Destructor
TCLstrbuff::TCLstrbuff(const string& name, const string& id, TCL* tcl):
  GuiVar(name, id, tcl),
  ostringstream()
{
  TCLTask::lock();
  
  buff_ = Tcl_Alloc(bSize_=256);
  *buff_='\0';
  
  if ( Tcl_LinkVar(the_interp, ccast_unsafe(varname_), (char*)&buff_, TCL_LINK_STRING | TCL_LINK_READ_ONLY)!=TCL_OK){
    cout << "Not possible to link tcl var" << std::endl;
  }
  
  TCLTask::unlock();
}

TCLstrbuff::~TCLstrbuff(){
  Tcl_UnlinkVar(the_interp, ccast_unsafe(varname_));
  Tcl_UnsetVar(the_interp, ccast_unsafe(varname_), TCL_GLOBAL_ONLY);
  Tcl_Free(buff_);
}

//////////
//   
void TCLstrbuff::emit(std::ostream&, string&){
  
}

//////////
//
TCLstrbuff& TCLstrbuff::flush(){
 
  TCLTask::lock();

  const int n = tellp();
  if (n>bSize_)
    buff_ = Tcl_Realloc(buff_, bSize_ = n);
  
  strcpy(buff_, ostringstream::str().c_str()); 
  Tcl_UpdateLinkedVar(the_interp, ccast_unsafe(varname_)); 
  TCLTask::unlock();
  
  // reinitializing the stream
  ostringstream::clear();
  ostringstream::seekp(0);
#ifdef __sgi
  ostringstream::str("");
#endif
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
