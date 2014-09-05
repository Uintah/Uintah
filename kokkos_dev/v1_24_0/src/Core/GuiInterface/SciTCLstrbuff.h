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
 *  TCLstrbuff.h: class to define string buffer for SciTclStream itcl-class 
 * 
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   January, 2001
 *   
 *   Copyright (C) 2000 SCI Group
 */

#ifndef TCL_OSTREAM_H
#define TCL_OSTREAM_H

#include <Core/GuiInterface/GuiVar.h>

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using namespace std;

class SCICORESHARE SciTCLstrbuff : public GuiVar, public ostringstream{
  
  // GROUP: private data
  //////////
  // Placeholder to C-type string for communication with Tcl
  char*      buff_;
  //////////
  // Size of allocated C-string
  int        bSize_;

public:
  
  // GROUP: Constructor/Destructor
  //////////
  // 
  SciTCLstrbuff(GuiContext* ctx);
  virtual ~SciTCLstrbuff();
  
  // GROUP: public member functions
  //////////
  // 
  SciTCLstrbuff& flush();
 
  template<class T> inline SciTCLstrbuff& operator<<(T pVal){
    static_cast<ostringstream&>(*this)<<pVal;
    return *this;
  }

  virtual void emit(std::ostream&, string& midx);
};

SciTCLstrbuff& operator<<(SciTCLstrbuff& stream, SciTCLstrbuff& (*mp)(SciTCLstrbuff&));
SciTCLstrbuff& endl(SciTCLstrbuff& stream);
SciTCLstrbuff& flush(SciTCLstrbuff& stream);
SciTCLstrbuff& ends(SciTCLstrbuff& stream);

} // end namespace SCIRun

#endif
