
/*
 *  ScalarFieldUG.cc: Scalar Fields defined on an unstructured grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <ScalarFieldUG.h>
#include <NotFinished.h>
#include <Classlib/String.h>

static Persistent* maker()
{
    return new ScalarFieldUG;
}

PersistentTypeID ScalarFieldUG::type_id("ScalarFieldUG", "ScalarField", maker);

ScalarFieldUG::ScalarFieldUG()
: ScalarField(UnstructuredGrid)
{
}

ScalarFieldUG::ScalarFieldUG(const MeshHandle& mesh)
: ScalarField(UnstructuredGrid), mesh(mesh), data(mesh->nodes.size())
{
}

ScalarFieldUG::~ScalarFieldUG()
{
}

ScalarField* ScalarFieldUG::clone()
{
    NOT_FINISHED("ScalarFieldUG::clone()");
    return 0;
}

void ScalarFieldUG::compute_bounds()
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

#define SCALARFIELDUG_VERSION 1

void ScalarFieldUG::io(Piostream& stream)
{
    int version=stream.begin_class("ScalarFieldUG", SCALARFIELDUG_VERSION);
    // Do the base class....
    ScalarField::io(stream);

    Pio(stream, mesh);
    Pio(stream, data);
}

void ScalarFieldUG::compute_minmax()
{
    if(have_minmax || data.size()==0)
	return;
    double min=data[0];
    double max=data[1];
    for(int i=0;i<data.size();i++){
	min=Min(min, data[i]);
	max=Max(max, data[i]);
    }
    data_min=min;
    data_max=max;
    have_minmax=1;
}

int ScalarFieldUG::interpolate(const Point& p, double& value)
{
    int ix;
    if(!mesh->locate(p, ix))
	return 0;
    double s1,s2,s3,s4;
    Element* e=mesh->elems[ix];
    mesh->get_interp(e, p, s1, s2, s3, s4);
    value=data[e->n1]*s1+data[e->n2]*s2+data[e->n3]*s3+data[e->n4]*s4;
    return 1;
}

Vector ScalarFieldUG::gradient(const Point& p)
{
    int ix;
    if(!mesh->locate(p, ix))
	return Vector(0,0,0);
    Vector g1, g2, g3, g4;
    Element* e=mesh->elems[ix];
    mesh->get_grad(e, p, g1, g2, g3, g4);
    return g1*data[e->n1]+g2*data[e->n2]+g3*data[e->n3]+g4*data[e->n4];
}
