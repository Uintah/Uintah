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
 *  SciTCLstrbuff.cc: implementation of string buffer for SciTclStream itcl-class 
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   January, 2000
 *   
 *   Copyright (C) 2000 SCI Group
 */

#include <Core/GuiInterface/SciTCLstrbuff.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <string.h>
#include <tcl.h>

extern "C" Tcl_Interp* the_interp;

namespace SCIRun {

using namespace std;

////////////
// SciTCLstrbuff implementation

//////////
// Constructor/Destructor
SciTCLstrbuff::SciTCLstrbuff(GuiContext* ctx)
 : GuiVar(ctx), ostringstream()
{
  ctx->dontSave();
  ctx->lock();
  
  buff_ = Tcl_Alloc(bSize_=4096);
  *buff_='\0';

  string varname = ctx->getfullname();
  if ( Tcl_LinkVar(the_interp, ccast_unsafe(varname), (char*)&buff_, TCL_LINK_STRING | TCL_LINK_READ_ONLY)!=TCL_OK){
    cout << "Not possible to link tcl var" << std::endl;
  }
  
  ctx->unlock();
}

SciTCLstrbuff::~SciTCLstrbuff(){
  string varname = ctx->getfullname();
  Tcl_UnlinkVar(the_interp, ccast_unsafe(varname));
  Tcl_UnsetVar(the_interp, ccast_unsafe(varname), TCL_GLOBAL_ONLY);
  Tcl_Free(buff_);
}

//////////
//   
void SciTCLstrbuff::emit(std::ostream&, string&){
  
}

//////////
//
SciTCLstrbuff& SciTCLstrbuff::flush(){
 
  ctx->lock();

  const string::size_type n = ostringstream::str().size();
  if (((int)n)>bSize_)
  {
    buff_ = Tcl_Realloc(buff_, bSize_ = n);
  }

  strcpy(buff_, ostringstream::str().c_str()); 
  string varname = ctx->getfullname();
  Tcl_UpdateLinkedVar(the_interp, ccast_unsafe(varname)); 

  ctx->unlock();
  
  // Reinitializing the stream.
  ostringstream::clear();
  ostringstream::str("");

  return *this;
}

SciTCLstrbuff& operator<<(SciTCLstrbuff& stream, SciTCLstrbuff& (*mp)(SciTCLstrbuff&)){
  return mp(stream);
}

SciTCLstrbuff& endl(SciTCLstrbuff& stream){
  static_cast<ostringstream&>(stream)<<'\n';
  stream.flush();
  return stream;
}

SciTCLstrbuff& flush(SciTCLstrbuff& stream){
  stream.flush();
  return stream;
}

SciTCLstrbuff& ends(SciTCLstrbuff& stream){
  static_cast<ostringstream&>(stream)<<'\0';
  stream.flush();
  return stream;
}

} // end namespace SCIRun
