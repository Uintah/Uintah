
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
#include <Geometry/BSphere.h>
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
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", " << p2 << ", " << p3 << ")" << endl;
	return;
    }
    normals.add(n);
    GeomVertexPrim::add(p1);
    GeomVertexPrim::add(p2);
    GeomVertexPrim::add(p3);
}

int GeomTriangles::size(void)
{
    return verts.size();
}

void GeomTriangles::add(const Point& p1, const Vector& v1,
			const Point& p2, const Vector& v2,
			const Point& p3, const Vector& v3) {
    Vector n(Cross(p2-p1, p3-p1));
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", v1, " << p2 << ", v2, " << p3 << ", v3)" << endl;
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
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", m1, " << p2 << ", m2, " << p3 << ", m3)" << endl;
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
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", c1, " << p2 << ", c2, " << p3 << ", c3)" << endl;
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
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", v1, m1, " << p2 << ", v2, m2, " << p3 << ", v3, m3)" << endl;
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
	cerr << "Degenerate triangle in GeomTriangles::add(v1->" << v1->p << ", v2->" << v2->p << ", v3->" << v3->p << ")" << endl;
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

GeomTrianglesP::GeomTrianglesP()
{
    // don't really need to do anythin...
}

GeomTrianglesP::~GeomTrianglesP()
{

}

int GeomTrianglesP::size(void)
{
    return points.size()/9;
}

int GeomTrianglesP::add(const Point& p1, const Point& p2, const Point& p3)
{
    Vector n(Cross(p2-p1, p3-p1));
    if(n.length2() > 0){
        n.normalize();
    }   	
    else {
	cerr << "degenerate triangle!!!\n" << endl;
	return 0;
    }

    normals.add(n.x());
    normals.add(n.y());
    normals.add(n.z());

    points.add(p1.x());
    points.add(p1.y());
    points.add(p1.z());

    points.add(p2.x());
    points.add(p2.y());
    points.add(p2.z());

    points.add(p3.x());
    points.add(p3.y());
    points.add(p3.z());

    return 1;
}

GeomObj* GeomTrianglesP::clone()
{
    // not implemented....
    return 0;
}

void GeomTrianglesP::get_bounds(BBox& box)
{
    for(int i=0;i<points.size();i+=3)
	box.extend(Point(points[i],points[i+1],points[i+2]));
}

void GeomTrianglesP::get_bounds(BSphere& box)
{
    for(int i=0;i<points.size();i+=3)
	box.extend(Point(points[i],points[i+1],points[i+2]));
}


void GeomTrianglesP::make_prims(Array1<GeomObj*>& /*free */,
				Array1<GeomObj*>& /*dontfree*/)
{
    // not implemented...
}

void GeomTrianglesP::preprocess()
{
    // not implemented...
}

void GeomTrianglesP::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomTrianglesP::intersect");
}

void GeomTrianglesP::io(Piostream&)
{
    // not implemented...
}

// PersistentTypeID GeomTrianglesP::type_id;

GeomTrianglesPC::GeomTrianglesPC()
{
    // don't really need to do anythin...
}

GeomTrianglesPC::~GeomTrianglesPC()
{

}

int GeomTrianglesPC::add(const Point& p1, const Color& c1,
			const Point& p2, const Color& c2,
			const Point& p3, const Color& c3)
{
    if (GeomTrianglesP::add(p1,p2,p3)) {
	colors.add(c1.r());
	colors.add(c1.g());
	colors.add(c1.b());

	colors.add(c2.r());
	colors.add(c2.g());
	colors.add(c2.b());

	colors.add(c3.r());
	colors.add(c3.g());
	colors.add(c3.b());
	return 1;
    }

    return 0;
}

void GeomTrianglesPC::io(Piostream&)
{
    // not implemented...
}

//PersistentTypeID GeomTrianglesPC::type_id;

