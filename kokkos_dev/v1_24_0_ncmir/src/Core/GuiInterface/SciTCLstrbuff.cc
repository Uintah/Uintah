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
