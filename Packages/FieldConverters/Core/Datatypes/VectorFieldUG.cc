
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

#include <FieldConverters/Core/Datatypes/VectorFieldUG.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>

namespace FieldConverters {

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
    data.resize(mesh->nodesize());
    break;
  case ElementValues:
    data.resize(mesh->elemsize());
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
    if(have_bounds || mesh->nodesize() == 0)
	return;
    mesh->get_bounds(bmin, bmax);
    have_bounds=1;
}

int VectorFieldUG::interpolate(const Point& p, Vector& value)
{
    int ix=0;
    if(!mesh->locate(&ix, p, 0)) return 0;
    if(typ == NodalValues){
	double s1,s2,s3,s4;
	Element* e=mesh->element(ix);
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
	if(!mesh->locate2(&ix, p, 0))
	    return 0;
    if (!exhaustive)
	if(!mesh->locate(&ix, p))
	    return 0;
    if(typ == NodalValues){
	double s1,s2,s3,s4;
	Element* e=mesh->element(ix);
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

    SCIRun::Pio(stream, mesh);
    SCIRun::Pio(stream, data);
    stream.end_class();
}

void VectorFieldUG::get_boundary_lines(Array1<Point>& lines)
{
    mesh->get_boundary_lines(lines);
}

} // End namespace FieldConverters
