
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

#include <SCICore/Geom/Pickable.h>
#include <SCICore/Util/NotFinished.h>

namespace SCICore {
namespace GeomSpace {

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
Pickable::geom_pick( GeomPick *, Roe *, int, const BState & )
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

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:20  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:50  mcq
// Initial commit
//
// Revision 1.2  1999/05/13 18:14:04  dav
// updated Pickable to use pure virtual functions
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

