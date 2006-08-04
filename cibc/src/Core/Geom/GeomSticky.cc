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
 *  Sticky.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Geom/GeomSticky.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

Persistent *make_GeomSticky() {
  return scinew GeomSticky( 0 );
}

PersistentTypeID GeomSticky::type_id("GeomSticky", "GeomObj", make_GeomSticky);

GeomSticky::GeomSticky( GeomHandle obj )
  : GeomContainer(obj)
{
}

GeomSticky::GeomSticky( const GeomSticky &copy )
  : GeomContainer(copy)
{
}

GeomSticky::~GeomSticky()
{
}

GeomObj* GeomSticky::clone() {
  return scinew GeomSticky( *this );
}

void
GeomSticky::get_bounds( BBox& bb )
{
  // Stick to screen, no bbox.
}

#define GeomSticky_VERSION 1

void
GeomSticky::io(Piostream& stream)
{
  stream.begin_class("GeomSticky", GeomSticky_VERSION);
  GeomObj::io(stream);
  Pio(stream, child_);
  stream.end_class();
}

} // End namespace SCIRun

