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
 *  Pickable.cc: ???
 *
 *  Written by:
 *   Dav de St. Germain...
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1999
 *
 *  Copyright (C) 1999 University of Utah
 */

#include <Core/GeomInterface/Pickable.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Util/NotFinished.h>

namespace SCIRun {

WidgetPickable::~WidgetPickable()
{
}

void
WidgetPickable::geom_pick(GeomPickHandle, ViewWindow*, int, const BState&)
{
}
 
void
WidgetPickable::geom_release(GeomPickHandle, int, const BState&)
{
}

void
WidgetPickable::geom_moved(GeomPickHandle, int, double, const Vector&,
			   int, const BState&, const Vector &)
{
}



ModulePickable::~ModulePickable()
{
}

void
ModulePickable::geom_pick(GeomPickHandle, void*, GeomHandle)
{
}

void
ModulePickable::geom_release(GeomPickHandle, void*, GeomHandle)
{
}

void
ModulePickable::geom_moved(GeomPickHandle, int, double, const Vector&,
			  void*, GeomHandle)
{
}




} // End namespace SCIRun


