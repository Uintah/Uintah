
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

#include <Datatypes/ScalarFieldUG.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>

static Persistent* maker()
{
    return scinew ScalarFieldUG;
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
    mesh->get_bounds(bmin, bmax);
    have_bounds=1;
}

#define SCALARFIELDUG_VERSION 1

void ScalarFieldUG::io(Piostream& stream)
{
    /*int version=*/stream.begin_class("ScalarFieldUG", SCALARFIELDUG_VERSION);
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

int ScalarFieldUG::interpolate(const Point& p, double& value, double epsilon1, double epsilon2)
{
    int ix=0;
    if(!mesh->locate(p, ix, epsilon1, epsilon2))
	return 0;
    double s1,s2,s3,s4;
    Element* e=mesh->elems[ix];
    mesh->get_interp(e, p, s1, s2, s3, s4);
    value=data[e->n[0]]*s1+data[e->n[1]]*s2+data[e->n[2]]*s3+data[e->n[3]]*s4;
    return 1;
}

int ScalarFieldUG::interpolate(const Point& p, double& value, int& ix, double epsilon1, double epsilon2)
{
    if(!mesh->locate(p, ix, epsilon1, epsilon2))
	return 0;
    double s1,s2,s3,s4;
    Element* e=mesh->elems[ix];
    mesh->get_interp(e, p, s1, s2, s3, s4);
    value=data[e->n[0]]*s1+data[e->n[1]]*s2+data[e->n[2]]*s3+data[e->n[3]]*s4;
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
    return g1*data[e->n[0]]+g2*data[e->n[1]]+g3*data[e->n[2]]+g4*data[e->n[3]];
}

#ifdef __GNUG__

#include <Classlib/Array1.cc>
template class Array1<double>;
template void Pio(Piostream&, Array1<double>&);

#endif


#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/Array1.cc>

static void _dummy_(Piostream& p1, Array1<double>& p2)
{
    Pio(p1, p2);
}

#endif
#endif

