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


#include <Core/Parts/GraphPart.h>
#include <Core/Parts/PartInterface.h>
#include <Core/PartsGui/PartGui.h>

#include <Core/PartsGui/GraphGui.h>

namespace SCIRun {

void
PartGui::add_child( PartInterface *child )
{
  static int n = 0;

  string type = child->type();

  if ( type == "GraphGui" ) {
    GraphGui *gui = new GraphGui( name_+"-c"+to_string(n++) );
    string child_window;
    tcl_eval( "new-child-window", child_window );
    gui->set_window( child_window ); 

    gui->attach( child );
  }
}

} // namespace SCIRun

