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

#include <Core/GuiInterface/GuiManager.h>
#include <Core/Parts/TCLstrbuff.h>
#include <Core/Containers/StringUtil.h>
#include <string.h>

namespace SCIRun {

using namespace std;

////////////
// TCLstrbuff implementation

//////////
// Constructor/Destructor
TCLstrbuff::TCLstrbuff(const string& name, const string& id, Part* tcl):
  GuiVar(name, id, tcl),
  ostringstream()
{
  //  gm->create_var( varname_ );
}

TCLstrbuff::~TCLstrbuff()
{
  // gm->remove_var( varname_ );
}

//////////
//   
void TCLstrbuff::emit(std::ostream&, string&){
  
}

//////////
//
TCLstrbuff& TCLstrbuff::flush()
{
  //  gm->set_var( varname_, ostringstream::str() );

  cerr << "TCLstrbuff: " << ostringstream::str() << endl;
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
