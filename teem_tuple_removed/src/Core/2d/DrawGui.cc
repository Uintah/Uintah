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
 *  DrawGui.cc: 
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <Core/Malloc/Allocator.h>
#include <Core/2d/DrawGui.h>

namespace SCIRun {


  //PersistentTypeID DrawGui::type_id("DrawGUI", "DrawObj", make_DrawGui);

DrawGui::DrawGui(GuiInterface* gui, const string &name, const string& script)
  : TclObj(gui, script), DrawObj(name)
{
}

void 
DrawGui::set_windows( const string &menu, const string &tb,
		      const string &ui, const string &ogl )
{
  menu_ = menu;
  tb_ = tb;
  ui_ = ui;
  
  string space(" ");
  TclObj::set_window( id(), 
		      menu+" "+tb+" "+ui+" "+ogl );
}

DrawGui::~DrawGui()
{
}

} // namespace SCIRun

  
