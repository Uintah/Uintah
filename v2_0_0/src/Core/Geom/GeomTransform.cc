/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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

GeomTransform::GeomTransform(GeomHandle obj)
: GeomContainer(obj)
{
}

GeomTransform::GeomTransform(GeomHandle obj, const Transform trans)
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

GeomObj* GeomTransform::clone()
{
    return scinew GeomTransform(*this);
}

void GeomTransform::get_bounds(BBox& bb)
{
    BBox b;
    child_->get_bounds(b);
    bb.extend(trans.project(b.min()));
    bb.extend(trans.project(b.max()));
}

#define GEOMTransform_VERSION 1

void
GeomTransform::io(Piostream& stream)
{
    stream.begin_class("GeomTransform", GEOMTransform_VERSION);
    GeomContainer::io(stream);
//    Pio(stream, trans);
    stream.end_class();
}


} // End namespace SCIRun



