/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/DifferenceGeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>

#include <vector>
#include <sstream>

using namespace SCIRun;
using namespace Uintah;

const string DifferenceGeometryPiece::TYPE_NAME = "difference";

DifferenceGeometryPiece::DifferenceGeometryPiece(ProblemSpecP &ps) 
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";
  std::vector<GeometryPieceP> objs;

  GeometryPieceFactory::create(ps,objs);
  
  //__________________________________
  // bulletproofing
  if (objs.size() != 2){
    std::ostringstream warn;
    warn <<  "\nERROR:Input File:Geom_object:Difference:  You need two geom_objects in order to take the difference of them"; 
    throw ProblemSetupException(warn.str(),__FILE__, __LINE__);  
  }

  left_  = objs[0];
  right_ = objs[1];

}

DifferenceGeometryPiece::DifferenceGeometryPiece(GeometryPieceP p1,
                                                 GeometryPieceP p2)
  : left_(p1), right_(p2)
{
  name_ = "Unnamed " + TYPE_NAME + " from pieces";
}

DifferenceGeometryPiece::~DifferenceGeometryPiece()
{
}

DifferenceGeometryPiece::DifferenceGeometryPiece(const DifferenceGeometryPiece& rhs)
{
  name_ = "Unnamed " + TYPE_NAME + " from CpyCnstr";

  left_  = rhs.left_->clone();
  right_ = rhs.right_->clone();
}

DifferenceGeometryPiece&
DifferenceGeometryPiece::operator=(const DifferenceGeometryPiece& rhs)
{
  if (this == &rhs)
    return *this;

  left_  = rhs.left_->clone();
  right_ = rhs.right_->clone();

  return *this;
}

void
DifferenceGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  left_->outputProblemSpec( ps );
  right_->outputProblemSpec( ps );
}

GeometryPieceP
DifferenceGeometryPiece::clone() const
{
  return scinew DifferenceGeometryPiece(*this);
}

bool
DifferenceGeometryPiece::inside(const Point &p) const 
{
  return (left_->inside(p) && !right_->inside(p));
}

Box
DifferenceGeometryPiece::getBoundingBox() const
{
   // Initialize the lo and hi points to the left element
  Point left_lo = left_->getBoundingBox().lower();
  Point left_hi = left_->getBoundingBox().upper();
  Point right_lo = right_->getBoundingBox().lower();
  Point right_hi = right_->getBoundingBox().upper();
   
  Point lo = Min(left_lo,right_lo);
  Point hi = Max(left_hi,right_hi);

  return Box(lo,hi);
}
