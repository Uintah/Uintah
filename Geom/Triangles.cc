
/*
 *  Triangles.cc: Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Geom/Triangles.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>
#include <Malloc/Allocator.h>

Persistent* make_GeomTriangles()
{
    return new GeomTriangles;
}

PersistentTypeID GeomTriangles::type_id("GeomTriangles", "GeomObj", make_GeomTriangles);

GeomTriangles::GeomTriangles()
{
}

GeomTriangles::GeomTriangles(const GeomTriangles& copy)
: GeomVertexPrim(copy)
{
}

GeomTriangles::~GeomTriangles() {
}

void GeomTriangles::add(const Point& p1, const Point& p2, const Point& p3) {
    Vector n(Cross(p2-p1, p3-p1));
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle!!!\n" << endl;
	return;
    }
    normals.add(n);
    GeomVertexPrim::add(p1);
    GeomVertexPrim::add(p2);
    GeomVertexPrim::add(p3);
}

void GeomTriangles::add(const Point& p1, const Vector& v1,
			const Point& p2, const Vector& v2,
			const Point& p3, const Vector& v3) {
    Vector n(Cross(p2-p1, p3-p1));
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle!!!\n" << endl;
	return;
    }
    normals.add(n);
    GeomVertexPrim::add(p1, v1);
    GeomVertexPrim::add(p2, v2);
    GeomVertexPrim::add(p3, v3);
}

void GeomTriangles::add(const Point& p1, const MaterialHandle& m1,
			const Point& p2, const MaterialHandle& m2,
			const Point& p3, const MaterialHandle& m3) {
    Vector n(Cross(p2-p1, p3-p1));
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle!!!\n" << endl;
	return;
    }
    normals.add(n);
    GeomVertexPrim::add(p1, m1);
    GeomVertexPrim::add(p2, m2);
    GeomVertexPrim::add(p3, m3);
}

void GeomTriangles::add(const Point& p1, const Color& c1,
			const Point& p2, const Color& c2,
			const Point& p3, const Color& c3) {
    Vector n(Cross(p2-p1, p3-p1));
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle!!!\n" << endl;
	return;
    }
    normals.add(n);
    GeomVertexPrim::add(p1, c1);
    GeomVertexPrim::add(p2, c2);
    GeomVertexPrim::add(p3, c3);
}

void GeomTriangles::add(const Point& p1, const Vector& v1, 
			const MaterialHandle& m1, const Point& p2, 
			const Vector& v2, const MaterialHandle& m2,
			const Point& p3, const Vector& v3, 
			const MaterialHandle& m3) {
    Vector n(Cross(p2-p1, p3-p1));
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle!!!\n" << endl;
	return;
    }
    normals.add(n);
    GeomVertexPrim::add(p1, v1, m1);
    GeomVertexPrim::add(p2, v2, m2);
    GeomVertexPrim::add(p3, v3, m3);
}

void GeomTriangles::add(GeomVertex* v1, GeomVertex* v2, GeomVertex* v3) {
    Vector n(Cross(v3->p - v1->p, v2->p - v1->p));
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle!!!\n" << endl;
	return;
    }
    normals.add(n);
    GeomVertexPrim::add(v1);
    GeomVertexPrim::add(v2);
    GeomVertexPrim::add(v3);
}

void GeomTriangles::make_prims(Array1<GeomObj*>&,
			      Array1<GeomObj*>&)
{
    NOT_FINISHED("GeomTriangles::make_prims");
}

GeomObj* GeomTriangles::clone()
{
    return scinew GeomTriangles(*this);
}

void GeomTriangles::preprocess()
{
    NOT_FINISHED("GeomTriangles::preprocess");
}

void GeomTriangles::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomTriangles::intersect");
}

#define GEOMTRIANGLES_VERSION 1

void GeomTriangles::io(Piostream& stream)
{
    stream.begin_class("GeomTriangles", GEOMTRIANGLES_VERSION);
    GeomVertexPrim::io(stream);
    Pio(stream, normals);
    stream.end_class();
}
