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

#include <Core/GeometryPiece/BoxGeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Point.h>
#include <sstream>
#ifndef DMIN
#  define DMIN(a,b) (a < b ? a : b)
#endif

using namespace Uintah;
using namespace SCIRun;

const string BoxGeometryPiece::TYPE_NAME = "box";

BoxGeometryPiece::BoxGeometryPiece( ProblemSpecP & ps )
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";

  Point min, max;
  ps->require("min",min);
  ps->require("max",max); 
  
  double near_zero = 1e-100;
  double xdiff =  max.x() - min.x();
  double ydiff =  max.y() - min.y();
  double zdiff =  max.z() - min.z();
  
  if ( xdiff < near_zero   ||
       ydiff < near_zero   ||
       zdiff < near_zero ) {
    std::ostringstream warn;
    warn << "Input File Error: box max " << max << " <= min " << min << " coordinates" ;
    SCI_THROW(ProblemSetupException(warn.str(), __FILE__, __LINE__));
  }

  d_box=Box(min,max);
}

BoxGeometryPiece::BoxGeometryPiece(const Point& p1, const Point& p2)
  : d_box(Min(p1, p2), Max(p1, p2))
{
  name_ = "Unnamed " + TYPE_NAME + " from p1,p2";

  if(d_box.degenerate())
    SCI_THROW(ProblemSetupException("degenerate box", __FILE__, __LINE__));
}

BoxGeometryPiece::~BoxGeometryPiece()
{
}

void
BoxGeometryPiece::outputHelper( ProblemSpecP& ps ) const
{
  ps->appendElement("min",d_box.lower());
  ps->appendElement("max",d_box.upper());
}

GeometryPieceP
BoxGeometryPiece::clone() const
{
  return scinew BoxGeometryPiece(*this);
}

bool
BoxGeometryPiece::inside(const Point& p) const
{
  //Check p with the lower coordinates

  if (p == Max(p,d_box.lower()) && p == Min(p,d_box.upper()) )
    return true;
  else
    return false;
               
}

Box
BoxGeometryPiece::getBoundingBox() const
{
  return d_box;
}

double 
BoxGeometryPiece::volume() const
{
  double dx = (d_box.upper()).x() - (d_box.lower()).x();
  double dy = (d_box.upper()).y() - (d_box.lower()).y();
  double dz = (d_box.upper()).z() - (d_box.lower()).z();
  return (dx*dy*dz);
}

double 
BoxGeometryPiece::smallestSide() const
{
  double dx = (d_box.upper()).x() - (d_box.lower()).x();
  double dy = (d_box.upper()).y() - (d_box.lower()).y();
  double dz = (d_box.upper()).z() - (d_box.lower()).z();
  return DMIN(DMIN(dx,dy),dz);
}

unsigned int 
BoxGeometryPiece::thicknessDirection() const
{
  double dx = (d_box.upper()).x() - (d_box.lower()).x();
  double dy = (d_box.upper()).y() - (d_box.lower()).y();
  double dz = (d_box.upper()).z() - (d_box.lower()).z();
  double min = DMIN(DMIN(dx,dy),dz);
  if (dx == min) return 0;
  else if (dy == min) return 1;
  return 2;
}
