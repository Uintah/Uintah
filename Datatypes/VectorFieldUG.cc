
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

#include <VectorFieldUG.h>
#include <Classlib/String.h>
#include <NotFinished.h>

static Persistent* maker()
{
    return new VectorFieldUG;
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
    Point min(mesh->nodes[0]->p);
    Point max(mesh->nodes[0]->p);
    for(int i=1;i<mesh->nodes.size();i++){
	min=Min(min, mesh->nodes[i]->p);
	max=Max(max, mesh->nodes[i]->p);
    }
    bmin=min;
    bmax=max;
    have_bounds=1;
}

int VectorFieldUG::interpolate(const Point&, Vector&)
{
    NOT_FINISHED("VectorFieldUG::interpolate");
    return 0;
}

#define VECTORFIELDUG_VERSION 1

void VectorFieldUG::io(Piostream& stream)
{
    int version=stream.begin_class("VectorFieldUG", VECTORFIELDUG_VERSION);
    // Do the base class....
    VectorField::io(stream);

    Pio(stream, mesh);
    Pio(stream, data);
}
