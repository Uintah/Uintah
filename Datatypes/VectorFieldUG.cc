
/*
 *  VectorFieldUG.h: Vector Fields defined on an unstructured grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/VectorFieldUG.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>

using sci::MeshHandle;
using sci::Element;

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

#ifdef __GNUG__

#include <Classlib/Array1.cc>
template class Array1<Vector>;
template void Pio(Piostream&, Array1<Vector>&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/Array1.cc>

static void _dummy_(Piostream& p1, Array1<Vector>& p2)
{
    Pio(p1, p2);
}

#endif
#endif

