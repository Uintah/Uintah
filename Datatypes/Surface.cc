/*
 *  Surface.cc: The Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/Surface.h>
#include <Classlib/NotFinished.h>
#include <Geometry/Grid.h>

PersistentTypeID Surface::type_id("Surface", "Datatype", 0);

Surface::Surface(Representation rep, int closed)
: rep(rep), grid(0), closed(closed), pntHash(0), boundary_type(None)
{
}

Surface::~Surface()
{
    destroy_grid();
    destroy_hash();
}

Surface::Surface(const Surface& copy)
: closed(copy.closed)
{
    NOT_FINISHED("Surface::Surface");
}

void Surface::destroy_grid()
{
    if (grid) delete grid;
}

void Surface::destroy_hash() {
    if (pntHash) delete pntHash;
}

#define SURFACE_VERSION 3

void Surface::io(Piostream& stream) {
    int version=stream.begin_class("Surface", SURFACE_VERSION);
    Pio(stream, name);
    if (version >= 3){
	int* btp=(int*)&boundary_type;
	Pio(stream, *btp);
	Pio(stream, boundary_expr);
    }
    if (version == 2) {
	Array1<double> conductivity;
	Pio(stream, conductivity);
	int bt;
	Pio(stream, bt);
    }
    stream.end_class();
}

TriSurface* Surface::getTriSurface()
{
    if(rep==TriSurf)
	return (TriSurface*)this;
    else
	return 0;
}

void Surface::set_bc(const clString& bc_expr)
{
    boundary_expr=bc_expr;
    boundary_type=DirichletExpression;
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>
template class LockingHandle<Surface>;

#endif
