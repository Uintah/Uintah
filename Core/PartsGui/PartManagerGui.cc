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






