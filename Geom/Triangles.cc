
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
#include <Geom/Save.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Malloc/Allocator.h>

Persistent* make_GeomTriangles()
{
    return scinew GeomTriangles;
}

PersistentTypeID GeomTriangles::type_id("GeomTriangles", "GeomObj", make_GeomTriangles);

Persistent* make_GeomTrianglesP()
{
    return scinew GeomTrianglesP;
}

PersistentTypeID GeomTrianglesP::type_id("GeomTrianglesP", "GeomObj", make_GeomTrianglesP);

Persistent* make_GeomTrianglesPC()
{
    return scinew GeomTrianglesPC;
}

PersistentTypeID GeomTrianglesPC::type_id("GeomTrianglesPC", "GeomTrianglesP", make_GeomTrianglesPC);

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
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", " << p2 << ", " << p3 << ")" << endl;
	return;
    }
#endif
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
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", v1, " << p2 << ", v2, " << p3 << ", v3)" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1, v1);
    GeomVertexPrim::add(p2, v2);
    GeomVertexPrim::add(p3, v3);
}

void GeomTriangles::add(const Point& p1, const MaterialHandle& m1,
			const Point& p2, const MaterialHandle& m2,
			const Point& p3, const MaterialHandle& m3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", m1, " << p2 << ", m2, " << p3 << ", m3)" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1, m1);
    GeomVertexPrim::add(p2, m2);
    GeomVertexPrim::add(p3, m3);
}

void GeomTriangles::add(const Point& p1, const Color& c1,
			const Point& p2, const Color& c2,
			const Point& p3, const Color& c3) {
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", c1, " << p2 << ", c2, " << p3 << ", c3)" << endl;
	return;
    }
#endif
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
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(" << p1 << ", v1, m1, " << p2 << ", v2, m2, " << p3 << ", v3, m3)" << endl;
	return;
    }
#endif
    normals.add(n);
    GeomVertexPrim::add(p1, v1, m1);
    GeomVertexPrim::add(p2, v2, m2);
    GeomVertexPrim::add(p3, v3, m3);
}

void GeomTriangles::add(GeomVertex* v1, GeomVertex* v2, GeomVertex* v3) {
    Vector n(Cross(v3->p - v1->p, v2->p - v1->p));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
	n.normalize();
    } else {
	cerr << "Degenerate triangle in GeomTriangles::add(v1->" << v1->p << ", v2->" << v2->p << ", v3->" << v3->p << ")" << endl;
	cerr << "Degenerate triangle!!!\n" << endl;
	return;
    }
#endif
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

bool GeomTriangles::saveobj(ostream&, const clString& format, GeomSave*)
{
    NOT_FINISHED("GeomTriangles::saveobj");
    return false;
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

void GeomTrianglesP::reserve_clear(int n)
{
    int np = points.size()/9;
    int delta = n - np;

    points.remove_all();
    normals.remove_all();

    if (delta > 0) {
	points.grow(delta);
	normals.grow(delta);
    }
	
}

int GeomTrianglesP::add(const Point& p1, const Point& p2, const Point& p3)
{
    Vector n(Cross(p2-p1, p3-p1));
#ifndef SCI_NORM_OGL
    if(n.length2() > 0){
        n.normalize();
    }   	
    else {
	cerr << "degenerate triangle!!!\n" << endl;
	return 0;
    }
#endif

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
    return new GeomTrianglesP(*this);
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
    NOT_FINISHED("GeomTrianglesP::make_prims");
}

void GeomTrianglesP::preprocess()
{
    NOT_FINISHED("GeomTrianglesP::preprocess");
}

void GeomTrianglesP::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomTrianglesP::intersect");
}

#define GEOMTRIANGLESP_VERSION 1

void GeomTrianglesP::io(Piostream& stream)
{
    stream.begin_class("GeomTrianglesP", GEOMTRIANGLESP_VERSION);
    GeomObj::io(stream);
    Pio(stream, points);
    Pio(stream, normals);
    stream.end_class();
}

bool GeomTrianglesP::saveobj(ostream&, const clString& format, GeomSave*)
{
    NOT_FINISHED("GeomTrianglesP::saveobj");
    return false;
}

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

#define GEOMTRIANGLESPC_VERSION 1

void GeomTrianglesPC::io(Piostream& stream)
{
    stream.begin_class("GeomTrianglesPC", GEOMTRIANGLESPC_VERSION);
    GeomTrianglesP::io(stream);
    Pio(stream, colors);
    stream.end_class();
}


bool GeomTrianglesPC::saveobj(ostream& out, const clString& format,
			      GeomSave* saveinfo)
{
    if(format == "vrml"){
	saveinfo->start_sep(out);
	saveinfo->start_node(out, "Coordinate3");
	saveinfo->indent(out);
	out << "point [";
	int np=points.size()/3;
	for(int i=0;i<np;i++){
	    if(i>0)
		out << ", ";
	    int idx=i*3;
	    out << " " << points[idx] << " " << points[idx+1] << " " << points[idx+2];
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "NormalBinding");
	saveinfo->indent(out);
	out << "value PER_VERTEX\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "Normal");
	saveinfo->indent(out);
	out << "vector [";
	for(i=0;i<np;i+=3){
	    if(i>0)
		out << ", ";
	    int idx=i;
	    out << " " << normals[idx] << " " << normals[idx+1] << " " << normals[idx+2];
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "MaterialBinding");
	saveinfo->indent(out);
	out << "value PER_VERTEX\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "Material");
	saveinfo->indent(out);
	out << "diffuseColor [";
	for(i=0;i<np;i++){
	    if(i>0)
		out << ", ";
	    int idx=i*3;
	    out << " " << colors[idx] << " " << colors[idx+1] << " " << colors[idx+2];
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->start_node(out, "IndexedFaceSet");
	saveinfo->indent(out);
	out << "coordIndex [";
	for(i=0;i<np;i++){
	    if(i>0)
		out << ", ";
	    out << " " << i;
	    if(i%3==2)
		out << ", -1";
	}
	out << " ]\n";
	saveinfo->end_node(out);
	saveinfo->end_sep(out);
	return true;
    } else {
	NOT_FINISHED("GeomTrianglesPC::saveobj");
	return false;
    }
}

