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

#include <SCICore/Geom/GeomTransform.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geom/GeomSave.h>
#include <SCICore/Malloc/Allocator.h>

#include <iostream>
using std::cerr;
using std::ostream;

namespace SCICore {
namespace GeomSpace {

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

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:46  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:27  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:15  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:46  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//


