
/*
 *  ContourSet.cc: The ContourSet Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/General/ContourSet.h>

#include <Core/Containers/String.h>
#include <Core/Datatypes/Surface.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
using std::cerr;
using std::endl;

#define Sqr(x) ((x)*(x))

namespace DaveW {
static Persistent* make_ContourSet(){
    return scinew ContourSet;
}

PersistentTypeID ContourSet::type_id("ContourSet", "Datatype", make_ContourSet);

Vector mmult(double *m, const Vector &v) {
    double x[3], y[3];
    x[0]=v.x();x[1]=v.y();x[2]=v.z();
    for (int i=0; i<3; i++) {
	y[i]=0;
	for (int j=0; j<3; j++) {
	    y[i]+=m[j*4+i]*x[j];
	}
    }
    return Vector(y[0],y[1],y[2]);
}

ContourSet::ContourSet()
{
    basis[0]=Vector(1,0,0);
    basis[1]=Vector(0,1,0);
    basis[2]=Vector(0,0,1);
    origin=Vector(0,0,0);
    space=1;
}

ContourSet::ContourSet(const ContourSet &copy)
: contours(copy.contours), conductivity(copy.conductivity),
  bdry_type(copy.bdry_type), origin(copy.origin), bbox(copy.bbox),
  space(copy.space), name(copy.name)
{
    basis[0]=copy.basis[0];
    basis[1]=copy.basis[1];
    basis[2]=copy.basis[2];
}

ContourSet::~ContourSet() {
}

void ContourSet::build_bbox() {
    if (bbox.valid()) return;
    for (int i=0; i<contours.size(); i++)
	for (int j=0; j<contours[i].size(); j++)
	    for (int k=0; k<contours[i][j].size(); k++)
		bbox.extend(contours[i][j][k]);
}

// translate will rebuild the bbox if it's invalid, and then translate
void ContourSet::translate(const Vector &v) {
    origin=origin+v;
    if (!bbox.valid())
	build_bbox();
    bbox.translate(v);
}

// sacle will rebuild the bbox if it's invalid, and then scale
void ContourSet::scale(double sc) {
    basis[0]=basis[0]*sc;
    basis[1]=basis[1]*sc;
    basis[2]=basis[2]*sc;
    if (!bbox.valid())
	build_bbox();
    bbox.scale(sc, origin);
}

// just takes the (dx, dy, dz) vector as input -- read off dials...
// rotate doesn't refigure bbox, it just sets it to invalid
void ContourSet::rotate(const Vector &rot) {
    Transform tran;
    tran.pre_rotate(rot.x(), Vector(1,0,0));
    tran.pre_rotate(rot.y(), Vector(0,1,0));
    tran.pre_rotate(rot.z(), Vector(0,0,1));
    double m[16];
    tran.get(m);
    for(int i=0;i<16;i++)
	cerr << "m[" << i << "]=" << m[i] << endl;
    basis[0]=tran.project(basis[0]);
    basis[1]=tran.project(basis[1]);
    basis[2]=tran.project(basis[2]);
    bbox.reset();
}

#define CONTOUR_SET_VERSION 3

void ContourSet::io(Piostream& stream) 
{
using namespace SCIRun;

    int version=stream.begin_class("ContourSet", CONTOUR_SET_VERSION);
    Pio(stream, levels);
    Pio(stream, level_map);
    Pio(stream, contours);
    if (version >=3) {
	Pio(stream, conductivity);
	Pio(stream, bdry_type);
    }
    Pio(stream, basis[0]);
    Pio(stream, basis[1]);
    Pio(stream, basis[2]);
    Pio(stream, origin);
    Pio(stream, space);
    if (version >= 2)
	Pio(stream, bbox);
    Pio(stream, name);
    Pio(stream, split_join);
    Pio(stream, matl);
    stream.end_class();
}

ContourSet* ContourSet::clone()
{
    return scinew ContourSet(*this);
} // End namespace DaveW
}



