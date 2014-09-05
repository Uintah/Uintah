//static char *id="@(#) $Id$";

/*
 *  VectorFieldZone.cc: A compound Vector field type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Datatypes/VectorFieldZone.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCICore {
namespace Datatypes {

static Persistent* maker()
{
    return scinew VectorFieldZone(0);
}

PersistentTypeID VectorFieldZone::type_id("VectorFieldZone", "VectorField", maker);

VectorFieldZone::VectorFieldZone(int nzones)
: VectorField(Zones), zones(nzones)
{
}

VectorFieldZone::~VectorFieldZone()
{
}

VectorField* VectorFieldZone::clone()
{
    NOT_FINISHED("VectorFieldZone::clone");
    return 0;
}

void VectorFieldZone::compute_bounds()
{
    using SCICore::Geometry::Min;
    using SCICore::Geometry::Max;

    if(zones.size()==0)
	return;
    zones[0]->get_bounds(bmin, bmax);
    for(int i=1;i<zones.size();i++){
	Point min, max;
	zones[i]->get_bounds(min, max);
	bmin=Min(min, bmin);
	bmax=Max(max, bmax);
    }
}

int VectorFieldZone::interpolate(const Point& p, Vector& v)
{
    for(int i=0;i<zones.size();i++){
	if(zones[i]->interpolate(p, v))
	    return 1;
    }
    return 0;
}

int VectorFieldZone::interpolate(const Point& p, Vector& v, int& cache, int ex)
{
    for(int i=0;i<zones.size();i++){
	if(zones[i]->interpolate(p, v, cache, ex))
	    return 1;
    }
    return 0;
}

#define VECTORFIELDZONE_VERSION 1

void VectorFieldZone::io(Piostream& stream)
{
#ifndef _WIN32
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    /* int version=*/
    stream.begin_class("VectorFieldZone", VECTORFIELDZONE_VERSION);

    Pio(stream, zones);
    stream.end_class();
#endif
}

void VectorFieldZone::get_boundary_lines(Array1<Point>& lines)
{
    for(int i=0;i<zones.size();i++)
	zones[i]->get_boundary_lines(lines);
}

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/09/23 01:07:07  moulding
// #ifndef'ed out the io functions, in win32, for these datatypes.  They are
// causing problems with Pio and namespaces in VC++.  Sooner or later these have
// to actually get fixed
//
// Revision 1.3  1999/08/25 03:48:46  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:39:00  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:33  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:47  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1  1999/04/25 04:07:22  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//
