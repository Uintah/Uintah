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
 *  GeomDL.cc: Create a display list for its child
 *
 *  Written by:
 *   Author Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Date July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Core/Geom/GeomDL.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <algorithm>

using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomDL()
{
  return scinew GeomDL(0);
}

PersistentTypeID GeomDL::type_id("GeomDL", "GeomObj", make_GeomDL);


GeomDL::GeomDL(GeomHandle obj)
  : GeomContainer(obj)
{
}

GeomDL::GeomDL(const GeomDL &copy)
  : GeomContainer(copy)
{
}


GeomObj*
GeomDL::clone()
{
  return scinew GeomDL(*this);
}


void
GeomDL::dl_register(DrawInfoOpenGL *info)
{
  drawinfo_.push_back(info);
}

void
GeomDL::dl_unregister(DrawInfoOpenGL *info)
{
  drawinfo_.erase(std::remove(drawinfo_.begin(), drawinfo_.end(), info),
		  drawinfo_.end());
}

#define GEOMDL_VERSION 1

void
GeomDL::io(Piostream& stream)
{

  /*int version=*/ stream.begin_class("GeomDL", GEOMDL_VERSION);
  Pio(stream, child_);
  stream.end_class();
}

} // End namespace SCIRun


