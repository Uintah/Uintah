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


//    File   : BoxClipper.cc
//    Author : Michael Callahan
//    Date   : September 2001

#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Point.h>
#include <Core/Datatypes/Clipper.h>


namespace SCIRun {

Clipper::~Clipper()
{
}

bool
Clipper::inside_p(const Point &/* p */)
{
  return false;
}

IntersectionClipper::IntersectionClipper(ClipperHandle c0, ClipperHandle c1)
  : clipper0_(c0),
    clipper1_(c1)
{
}


bool
IntersectionClipper::inside_p(const Point &p)
{
  // Use boolean OR, is shortcutting.
  return !(!clipper0_->inside_p(p) || !clipper1_->inside_p(p));
}

UnionClipper::UnionClipper(ClipperHandle c0, ClipperHandle c1)
  : clipper0_(c0),
    clipper1_(c1)
{
}

bool
UnionClipper::inside_p(const Point &p)
{
  return clipper0_->inside_p(p) || clipper1_->inside_p(p);
}

InvertClipper::InvertClipper(ClipperHandle c)
  : clipper_(c)
{
}


bool
InvertClipper::inside_p(const Point &p)
{
  return !clipper_->inside_p(p);
}

BoxClipper::BoxClipper(Transform &t)
  : trans_(t)
{
}


bool
BoxClipper::inside_p(const Point &p)
{
  Point ptrans;
  ptrans = trans_.project(p);
  return (ptrans.x() >= -1.0 && ptrans.x() < 1.0 &&
	  ptrans.y() >= -1.0 && ptrans.y() < 1.0 &&
	  ptrans.z() >= -1.0 && ptrans.z() < 1.0);
}

} // end namespace SCIRun

