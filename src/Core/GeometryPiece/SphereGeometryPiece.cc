/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/GeometryPiece/SphereGeometryPiece.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Box.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
using namespace SCIRun;

const string SphereGeometryPiece::TYPE_NAME = "sphere";

SphereGeometryPiece::SphereGeometryPiece(ProblemSpecP& ps)
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";

  Point orig = Point(0.,0.,0.);
  double rad = 0.;

  if(!ps->get("center", orig)) // Alternate specification
    ps->require("origin",orig);
  ps->require("radius",rad);
  
  if ( rad <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere radius must be > 0.0", __FILE__, __LINE__));
  
  d_origin = orig;
  d_radius = rad;
}

SphereGeometryPiece::SphereGeometryPiece(const Point& origin,
                                         double radius)
{
  if ( radius <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere radius must be > 0.0", __FILE__, __LINE__));
  
  d_origin = origin;
  d_radius = radius;
}

SphereGeometryPiece::~SphereGeometryPiece()
{
}

void
SphereGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendElement("origin",d_origin);
  ps->appendElement("radius",d_radius);
}

GeometryPieceP
SphereGeometryPiece::clone() const
{
  return scinew SphereGeometryPiece(*this);
}

bool
SphereGeometryPiece::inside(const Point& p) const
{
  Vector diff = p - d_origin;

  if (diff.length2() > d_radius*d_radius)
    return false;
  else 
    return true;
  
}

Box SphereGeometryPiece::getBoundingBox() const
{
    Point lo(d_origin.x()-d_radius,d_origin.y()-d_radius,
           d_origin.z()-d_radius);

    Point hi(d_origin.x()+d_radius,d_origin.y()+d_radius,
           d_origin.z()+d_radius);

    return Box(lo,hi);

}
