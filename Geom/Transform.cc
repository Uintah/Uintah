/*
 *  Transform.cc: Transform properties for Geometry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 *
 */

#include <Geom/Transform.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Geom/Save.h>
#include <Malloc/Allocator.h>

#include <iostream.h>

Persistent* make_GeomTransform()
{
    Transform t;
    return new GeomTransform(0, t);
}

PersistentTypeID GeomTransform::type_id("GeomTransform", "GeomObj", make_GeomTransform);

GeomTransform::GeomTransform(GeomObj* obj)
: GeomContainer(obj)
{
}

GeomTransform::GeomTransform(GeomObj* obj, const Transform trans)
: GeomContainer(obj), trans(trans)
{
}

GeomTransform::GeomTransform(const GeomTransform& copy)
: GeomContainer(copy), trans(copy.trans)
{
}

void GeomTransform::setTransform(const Transform copy) 
{
    trans=copy;
}

void GeomTransform::scale(const Vector &v) {
    trans.pre_scale(v);
}

void GeomTransform::translate(const Vector &v) {
    trans.pre_translate(v);
}

void GeomTransform::rotate(double angle, const Vector &v) {
    trans.pre_rotate(angle,v);
}

Transform GeomTransform::getTransform()
{
    return trans;
}

GeomTransform::~GeomTransform()
{
}

GeomObj* GeomTransform::clone()
{
    return scinew GeomTransform(*this);
}

void GeomTransform::get_bounds(BBox& bb)
{
    BBox b;
    child->get_bounds(b);
    bb.extend(trans.project(b.min()));
    bb.extend(trans.project(b.max()));
}

void GeomTransform::get_bounds(BSphere& bs)
{
    BBox b;
    child->get_bounds(b);
    BBox b2;
    b2.extend(trans.project(b.min()));
    b2.extend(trans.project(b.max()));
    bs.extend(b2.center(), b2.longest_edge()/1.9999999);
}

void GeomTransform::make_prims(Array1<GeomObj*>& free,
			      Array1<GeomObj*>& dontfree)
{
    child->make_prims(free, dontfree);
}

void GeomTransform::intersect(const Ray& ray, Material* matl, Hit& hit)
{
    child->intersect(ray, matl, hit);
}

#define GEOMTransform_VERSION 1

void GeomTransform::io(Piostream& stream)
{
    stream.begin_class("GeomTransform", GEOMTransform_VERSION);
    GeomContainer::io(stream);
//    Pio(stream, trans);
    stream.end_class();
}

bool GeomTransform::saveobj(ostream&, const clString&,
			   GeomSave*)
{
    cerr << "don't know how to output a transform matrix!\n";
    return false;

#if 0
    if(format == "vrml" || format == "iv"){
	saveinfo->start_sep(out);
	saveinfo->start_node(out, "Transform");
	saveinfo->indent(out);

	// not sure what to put here!

	saveinfo->end_node(out);
	if(!child->saveobj(out, format, saveinfo))
	    return false;
	saveinfo->end_sep(out);
	return true;
    } else if(format == "rib"){
	saveinfo->start_attr(out);
	saveinfo->indent(out);

	// not sure what to put here!

	if(!child->saveobj(out, format, saveinfo))
	    return false;
	saveinfo->end_attr(out);
	return true;
    } else {
	NOT_FINISHED("GeomTransform::saveobj");
	return false;
    }
#endif
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template class LockingHandle<Transform>;

#include <Classlib/Array1.cc>
template class Array1<TransformHandle>;

template void Pio(Piostream&, Array1<TransformHandle>&);
template void Pio(Piostream&, TransformHandle&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/LockingHandle.cc>

static void _dummy_(Piostream& p1, TransformHandle& p2)
{
    Pio(p1, p2);
}

#endif
#endif

