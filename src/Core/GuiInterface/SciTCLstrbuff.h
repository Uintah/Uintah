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
