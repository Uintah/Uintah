
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
 *  Axes.cc: 2D graph axes
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   August 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Malloc/Allocator.h>
#include <Core/2d/Axes.h>

namespace SCIRun {
 
Persistent* make_Axes()
{
  return scinew Axes;
}
 
PersistentTypeID Axes::type_id("axes", "Drawable", make_Axes);

Axes::~Axes()
{
}

void
Axes::set_color( const Color &c )
{
  color = c;
}

void
Axes::get_bounds( BBox2d & )
{
}

#define AXES_VERSION 1
 
void
Axes::io(Piostream& )
{
}

}
