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

#include <Core/Datatypes/Surface.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geometry/Grid.h>

namespace SCIRun {


PersistentTypeID Surface::type_id("Surface", "Datatype", 0);

Surface::Surface(Representation rep, int closed)
: monitor("Surface crowd monitor"), rep(rep), grid(0), closed(closed), pntHash(0), boundary_type(BdryNone)
{
}

Surface::~Surface()
{
    destroy_grid();
    destroy_hash();
}

Surface::Surface(const Surface& copy)
  : monitor("Surface crowd monitor"), closed(copy.closed), rep(copy.rep), name(copy.name), grid(0), pntHash(0)
{
//    NOT_FINISHED("Surface::Surface");
}

void Surface::destroy_grid()
{
    if (grid) delete grid;
}

void Surface::destroy_hash() {
    if (pntHash) delete pntHash;
}

#define SURFACE_VERSION 4

void Surface::io(Piostream& stream) {


    int version=stream.begin_class("Surface", SURFACE_VERSION);
    Pio(stream, name);
    if (version >= 4){
	int* repp=(int*)&rep;
	Pio(stream, *repp);
	int* btp=(int*)&boundary_type;
	Pio(stream, *btp);
	Pio(stream, boundary_expr);
    }	
    if (version == 3){
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

SurfTree* Surface::getSurfTree() {
    if (rep==STree)
	return (SurfTree*) this;
    else
	return 0;
}

TriSurface* Surface::getTriSurface()
{
    if(rep==TriSurf)
	return (TriSurface*)this;
    else
	return 0;
}

PointsSurface* Surface::getPointsSurface() {
    if (rep==PointsSurf)
	return (PointsSurface*)this;
    else
	return 0;
}

void Surface::set_bc(const clString& bc_expr)
{
    boundary_expr=bc_expr;
    boundary_type=DirichletExpression;
}

} // End namespace SCIRun

