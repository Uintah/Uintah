/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
    BBox b1;
    b1.extend(trans.project(b.min()));
    b1.extend(trans.project(b.max()));
    bb.extend(b1);
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



