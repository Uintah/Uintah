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
 *  PartManagerGui.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <iostream>

#include <Core/Util/Signals.h>
#include <Core/Parts/PartManager.h>
#include <Core/PartsGui/PartManagerGui.h>

namespace SCIRun {

  using std::cerr;
  using std::endl;

PartManagerGui::PartManagerGui( const string &name, const string &script)
  : PartGui( name, script )
{
  set_id(name);
}
 
PartManagerGui::~PartManagerGui()
{
}

void
PartManagerGui::add_part( const string &name )
{
  cerr << "PartManagerGui::add_part " << name << endl;
  tcl_ << "add-part " << name;
  tcl_exec();
}

void
PartManagerGui::select_part( int i )
{
  cerr << "PartManagerGui::select_part " << i << endl;
}

void 
PartManagerGui::attach( PartInterface *interface )
{
  PartManager *manager = dynamic_cast<PartManager *>(interface);
  if ( !manager ) {
    cerr << "PartManagerGui[connect]: got the wrong interface type\n";
    return;
  }

  connect( manager->has_part, this, &PartManagerGui::add_part);
  connect( manager->part_selected, this, &PartManagerGui::select_part);
  connect( part_selected, manager, &PartManager::select_part);
}


} // namespace MIT






