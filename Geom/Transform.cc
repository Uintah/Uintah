
/*
 *  Transform.cc: Transform properties for Geometry
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */




/****************   WARNING!!!!!!   *******************/
/*****  I didn't get a chance to verify that **********/
/*****  this actually works.  the rendering  **********/
/*****  still needs to be checked.  i didn't **********/
/*****  end up using this code, so it was    **********/
/*****  never checked.  sorry...             **********/
/****************   WARNING!!!!!!  ********************/






#include <Geom/Transform.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Geom/Save.h>
#include <Malloc/Allocator.h>

Persistent* make_GeomTransform()
{
    Transform t;
    return new GeomTransform(0, t);
}

PersistentTypeID GeomTransform::type_id("GeomTransform", "GeomObj", make_GeomTransform);

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

bool GeomTransform::saveobj(ostream& out, const clString& format,
			   GeomSave* saveinfo)
{
    cerr << "don't know how to output a transform matrix!\n";
    return false;

#if 0
    if(format == "vrml"){
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

