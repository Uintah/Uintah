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
 * Billboard.cc: Pts objects
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Geom/GeomBillboard.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomBillboard()
{
    return scinew GeomBillboard(0, Point(0,0,0));
}

PersistentTypeID GeomBillboard::type_id("GeomBillboard", "GeomObj",
					make_GeomBillboard);


GeomBillboard::GeomBillboard(GeomHandle obj, const Point &p)
: GeomContainer(obj), at(p)
{

}

GeomObj* GeomBillboard::clone()
{
  cerr << "GeomBillboard::clone not implemented!\n";
  return 0;
}

void GeomBillboard::get_bounds(BBox& box)
{
  BBox bbox;

  child_->get_bounds(bbox);

  box.reset();
  box.extend( Point( bbox.min().x(), bbox.min().z(), bbox.min().y() ));
  box.extend( Point( bbox.max().x(), bbox.max().z(), bbox.max().y() ));
  box.translate( at.vector() );
}

#define GEOMBBOXCACHE_VERSION 1

void GeomBillboard::io(Piostream& stream)
{

    stream.begin_class("GeomBillboard", GEOMBBOXCACHE_VERSION);
    Pio(stream, child_);
    stream.end_class();
}

} // End namespace SCIRun


