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

#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/BBox.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/String.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

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

} // End namespace SCIRun



