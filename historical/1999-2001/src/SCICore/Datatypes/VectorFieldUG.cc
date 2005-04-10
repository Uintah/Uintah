//static char *id="@(#) $Id$";

/*
 *  VectorFieldUG.cc: Vector Fields defined on an unstructured grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Datatypes/VectorFieldUG.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCICore {
namespace Datatypes {

static Persistent* maker()
{
    return scinew VectorFieldUG(VectorFieldUG::NodalValues);
}

PersistentTypeID VectorFieldUG::type_id("VectorFieldUG", "VectorField", maker);

VectorFieldUG::VectorFieldUG(Type typ)
: VectorField(UnstructuredGrid), typ(typ)
{
}

VectorFieldUG::VectorFieldUG(const MeshHandle& mesh, Type typ)
: VectorField(UnstructuredGrid), mesh(mesh),
  typ(typ)
{
  switch(typ){
  case NodalValues:
    data.resize(mesh->nodes.size());
    break;
  case ElementValues:
    data.resize(mesh->elems.size());
    break;
  }
}

VectorFieldUG::~VectorFieldUG()
{
}

VectorField* VectorFieldUG::clone()
{
    NOT_FINISHED("VectorFieldUG::clone()");
    return 0;
}

void VectorFieldUG::compute_bounds()
{
    if(have_bounds || mesh->nodes.size() == 0)
	return;
    mesh->get_bounds(bmin, bmax);
    have_bounds=1;
}

int VectorFieldUG::interpolate(const Point& p, Vector& value)
{
    int ix=0;
    if(!mesh->locate(p, ix, 0)) return 0;
    if(typ == NodalValues){
	double s1,s2,s3,s4;
	Element* e=mesh->elems[ix];
	mesh->get_interp(e, p, s1, s2, s3, s4);
	value=data[e->n[0]]*s1+data[e->n[1]]*s2+data[e->n[2]]*s3+data[e->n[3]]*s4;
    } else {
	value=data[ix];
    }
    return 1;
}

int VectorFieldUG::interpolate(const Point& p, Vector& value, int& ix, int exhaustive)
{
    if (exhaustive)
	if(!mesh->locate2(p, ix, 0))
	    return 0;
    if (!exhaustive)
	if(!mesh->locate(p, ix))
	    return 0;
    if(typ == NodalValues){
	double s1,s2,s3,s4;
	Element* e=mesh->elems[ix];
	mesh->get_interp(e, p, s1, s2, s3, s4);
	value=data[e->n[0]]*s1+data[e->n[1]]*s2+data[e->n[2]]*s3+data[e->n[3]]*s4;
    } else {
	value=data[ix];
    }
    return 1;
}

#define VECTORFIELDUG_VERSION 2

void VectorFieldUG::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    int version=stream.begin_class("VectorFieldUG", VECTORFIELDUG_VERSION);
    // Do the base class....
    VectorField::io(stream);

    if(version < 2){
	typ=NodalValues;
    } else {
	int* typp=(int*)&typ;
	stream.io(*typp);
    }

    Pio(stream, mesh);
    Pio(stream, data);
    stream.end_class();
}

void VectorFieldUG::get_boundary_lines(Array1<Point>& lines)
{
    mesh->get_boundary_lines(lines);
}

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:46  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:59  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:32  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:47  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/04/27 21:14:31  dav
// working on Datatypes
//
// Revision 1.2  1999/04/25 04:14:48  dav
// oopps...?
//
// Revision 1.1  1999/04/25 04:07:22  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

