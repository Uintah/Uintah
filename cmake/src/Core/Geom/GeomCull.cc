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

#include <Core/Geom/GeomCull.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomCull()
{
  return scinew GeomCull(0,0);
}
  
PersistentTypeID GeomCull::type_id("GeomCull", "GeomObj", make_GeomCull);

GeomCull::GeomCull(GeomHandle child, Vector *normal) :
  GeomContainer(child), normal_(0) 
{
  if (normal) normal_ = scinew Vector(*normal);
}

GeomCull::GeomCull(const GeomCull &copy) :
  GeomContainer(copy), normal_(0) 
{
  if (copy.normal_) normal_ = scinew Vector(*copy.normal_);
}

GeomObj *
GeomCull::clone() 
{
  return scinew GeomCull(*this);
}
  
void
GeomCull::set_normal(Vector *normal) {
  if (normal_) {
    delete normal_;
    normal_ = 0;
  }
  
  if (normal) {
    normal_ = scinew Vector(*normal);
  }
}

void
GeomCull::io(Piostream&stream) {
    stream.begin_class("GeomCull", 1);
    GeomContainer::io(stream); // Do the base class first...
    if (normal_) { 
      Pio(stream,*normal_);
    }
    stream.end_class();
  }

}
