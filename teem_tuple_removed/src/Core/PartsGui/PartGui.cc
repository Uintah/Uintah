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


