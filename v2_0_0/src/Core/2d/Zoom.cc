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
 *  Zoom.cc: 
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <Core/Datatypes/Color.h>
#include <Core/Malloc/Allocator.h>
#include <Core/2d/Zoom.h>
#include <Core/2d/Diagram.h>
#include <Core/GuiInterface/GuiInterface.h>

using namespace SCIRun;

Persistent* make_Zoom()
{
  return scinew Zoom(GuiInterface::getSingleton());
}

PersistentTypeID Zoom::type_id("Zoom", "BoxObj", make_Zoom);

 Zoom::Zoom(GuiInterface* gui, Diagram *, const string &name)
  : TclObj(gui, "Zoom" ), BoxObj(name)
{
}


Zoom::~Zoom()
{
}



#define ZOOM_VERSION 1

void 
Zoom::io(Piostream& stream)
{
  stream.begin_class("Zoom", ZOOM_VERSION);
  BoxObj::io(stream);
  stream.end_class();
}
