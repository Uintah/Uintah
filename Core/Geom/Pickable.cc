
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


