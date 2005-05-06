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
 *  PartGui.cc
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

#include <Core/Parts/GraphPart.h>
#include <Core/Parts/PartInterface.h>
#include <Core/PartsGui/PartGui.h>

#include <Core/PartsGui/GraphGui.h>
#include <Core/PartsGui/PartManagerGui.h>
#include <Core/PartsGui/NullGui.h>

namespace SCIRun {
using std::cerr;
using std::endl;

map<string,GuiCreatorBase *> PartGui::table;


GuiCreatorBase::GuiCreatorBase( const string &name )
{
  PartGui::table[name] = this;
}


static GuiCreator<GraphGui> graph_creator("GraphGui");
static GuiCreator<PartManagerGui> part_manager_creator("PartManager");
static GuiCreator<NullGui> null_creator("");

void
PartGui::add_child( PartInterface *child )
{
  string type = child->type();

  map<string,GuiCreatorBase *>::iterator creator = table.find(type);

  PartGui *gui;
  if ( creator == table.end() ) {
    gui = scinew NullGui( name_+"-c"+to_string(n_++) );
  }
  else 
    gui = creator->second->create( name_+"-c"+to_string(n_++) );

  if ( gui ) {
    string child_window;
    tcl_eval( "new-child-window "+ child->name(), child_window );
    gui->set_window( child_window ); 
    gui->attach( child );

    child->report_children( gui, &PartGui::add_child );
  }
  else
    cerr << "unknown gui\n";
}

} // namespace SCIRun


