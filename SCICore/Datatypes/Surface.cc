//static char *id="@(#) $Id$";

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

#include <SCICore/CoreDatatypes/Surface.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Geometry/Grid.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Geometry::Grid;

PersistentTypeID Surface::type_id("Surface", "Datatype", 0);

Surface::Surface(Representation rep, int closed)
: rep(rep), grid(0), closed(closed), pntHash(0), boundary_type(BdryNone)
{
}

Surface::~Surface()
{
    destroy_grid();
    destroy_hash();
}

Surface::Surface(const Surface& copy)
: closed(copy.closed), rep(copy.rep), name(copy.name), grid(0)
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

    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

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

TopoSurfTree* Surface::getTopoSurfTree() {
    if (rep==TSTree)
	return (TopoSurfTree*) this;
    else
	return 0;
}

SurfTree* Surface::getSurfTree() {
    if (rep==STree)
	return (SurfTree*) this;
    else
	return 0;
}

ScalarTriSurface* Surface::getScalarTriSurface()
{
    if(rep==ScalarTriSurf)
	return (ScalarTriSurface*)this;
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

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:55  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:29  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:45  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1  1999/04/27 21:14:29  dav
// working on CoreDatatypes
//
// Revision 1.2  1999/04/25 04:14:44  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//
