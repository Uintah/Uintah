
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

static Persistent* maker()
{
    return scinew VectorFieldUG;
}

PersistentTypeID VectorFieldUG::type_id("VectorFieldUG", "VectorField", maker);

VectorFieldUG::VectorFieldUG()
: VectorField(UnstructuredGrid)
{
}

VectorFieldUG::VectorFieldUG(const MeshHandle& mesh)
: VectorField(UnstructuredGrid), mesh(mesh), data(mesh->nodes.size())
{
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
    int ix;
    if(!mesh->locate(p, ix))
	return 0;
    double s1,s2,s3,s4;
    Element* e=mesh->elems[ix];
    mesh->get_interp(e, p, s1, s2, s3, s4);
    value=data[e->n[0]]*s1+data[e->n[1]]*s2+data[e->n[2]]*s3+data[e->n[3]]*s4;
    return 1;
}

#define VECTORFIELDUG_VERSION 1

void VectorFieldUG::io(Piostream& stream)
{
    /*int version=*/stream.begin_class("VectorFieldUG", VECTORFIELDUG_VERSION);
    // Do the base class....
    VectorField::io(stream);

    Pio(stream, mesh);
    Pio(stream, data);
}
