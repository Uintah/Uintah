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

#include <Core/Geom/Pickable.h>
#include <Core/Util/NotFinished.h>

namespace SCIRun {

Pickable::~Pickable()
{
}

#if 0
// These are now all pure virtual functions, so these are not needed...
void
Pickable::geom_pick( GeomPick *, void *, int )
{
  NOT_FINISHED( "This is a virtual function... should've been overloaded" );
}

void
Pickable::geom_pick( GeomPick *, void * )
{
  NOT_FINISHED( "This is a virtual function... should've been overloaded" );
}

void
Pickable::geom_pick( GeomPick *, ViewWindow *, int, const BState & )
{
  NOT_FINISHED( "This is a virtual function... should've been overloaded" );
}

void
Pickable::geom_release( GeomPick *, void *, int )
{
  NOT_FINISHED( "This is a virtual function... should've been overloaded" );
}

void
Pickable::geom_release( GeomPick *, void * )
{
  NOT_FINISHED( "This is a virtual function... should've been overloaded" );
}

void
Pickable::geom_release( GeomPick *, int, const BState & )
{
  NOT_FINISHED( "This is a virtual function... should've been overloaded" );
}

void
Pickable::geom_moved( GeomPick *, int, double, const Vector &,  void * )
{
  NOT_FINISHED( "This is a virtual function... should've been overloaded" );
}

void
Pickable::geom_moved(GeomPick*, int, double, const Vector&, void*, int)
{
  NOT_FINISHED( "This is a virtual function... should've been overloaded" );
}

void
Pickable::geom_moved(GeomPick*,int,double,const Vector&,int, const BState&)
{
  NOT_FINISHED( "This is a virtual function... should've been overloaded" );
}

void
Pickable::geom_moved(GeomPick*,int,double,const Vector&,const BState&, int)
{
  NOT_FINISHED( "This is a virtual function... should've been overloaded" );
}
#endif

} // End namespace SCIRun


