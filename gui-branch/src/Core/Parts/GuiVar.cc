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
 *  GuiVar.cc: Interface to TCL variables
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Changes for distributed Dataflow:
 *   Michelle Miller 
 *   Thu May 14 01:24:12 MDT 1998
 * FIX: error cases and GuiVar* get()
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Parts/GuiVar.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <iostream>
using std::cerr;
using std::endl;
using std::ostream;


namespace SCIRun {

GuiVar::GuiVar(const string& name, const string& id, Part *part)
  : varname_(name), is_reset_(1), part(part), id_(id)
{
}

GuiVar::~GuiVar()
{
}

void 
GuiVar::reset()
{
  is_reset_=1;
}

string 
GuiVar::str()
{
  return varname_;
}

string 
GuiVar::format_varname() 
{
  int state=0;
  int end_of_modulename = -1;
  //int space = 0;
  for (unsigned int i=0; i<varname_.size(); i++) {
    if (varname_[i] == ' ') return "unused";
    if (state == 0 && varname_[i] == '_') state=1;
    else if (state == 1 && isdigit(varname_[i])) state = 2;
    else if (state == 2 && isdigit(varname_[i])) state = 2;
    else if (state == 2 && varname_[i] == '-') {
      end_of_modulename = i;
      state = 0;
    } else state = 0;
  }
  if (end_of_modulename == -1)
    cerr << "Error -- couldn't format name "<< varname_ << endl;
  return varname_.substr(end_of_modulename+1);
}

template class GuiTriple<Point>;
template class GuiTriple<Vector>;

} // End namespace SCIRun


















