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
 *  Widget.cc: 
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
#include <Core/2d/Widget.h>


namespace SCIRun {

Persistent* make_Widget()
{
  return scinew Widget;
}

PersistentTypeID Widget::type_id("Widget", "DrawObj", make_Widget);

Widget::Widget( const string &name)
  : DrawObj(name)
{
}


Widget::~Widget()
{
}

  
#define WIDGET_VERSION 1

void 
Widget::io(Piostream& stream)
{
  stream.begin_class("Widget", WIDGET_VERSION);
  DrawObj::io(stream);
  stream.end_class();
}


} // namespace SCIRun

  
