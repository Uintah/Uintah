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


#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace SCIRun;
using namespace Uintah;

using std::vector;

const string UnionGeometryPiece::TYPE_NAME = "union";

UnionGeometryPiece::UnionGeometryPiece(ProblemSpecP& ps) 
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";
  // Need to loop through all the geometry pieces
  GeometryPieceFactory::create(ps,child_);
  
}

UnionGeometryPiece::UnionGeometryPiece(const vector<GeometryPieceP>& child) :
  child_(child)
{
  name_ = "Unnamed " + TYPE_NAME + " from vector";
}

UnionGeometryPiece&
UnionGeometryPiece::operator=(const UnionGeometryPiece& rhs){
  if (this == &rhs)
    return *this;

  child_.clear();

  // Copy in the new values
  for (vector<GeometryPieceP>::const_iterator it = rhs.child_.begin();
       it != rhs.child_.end(); ++it)
    child_.push_back((*it)->clone());

  return *this;
}

void
UnionGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  // If this is a named object, then only output the children the first time.
  for( vector<GeometryPieceP>::const_iterator it = child_.begin(); it != child_.end(); ++it ) {
    (*it)->outputProblemSpec( ps );
  }
}


GeometryPieceP
UnionGeometryPiece::clone() const
{
  return scinew UnionGeometryPiece(*this);
}

bool
UnionGeometryPiece::inside(const Point &p) const 
{
  for (int i = 0; i < (int)child_.size(); i++) {
    if (child_[i]->inside(p)) {
      return true;
    }
  }
  return false;
}

Box UnionGeometryPiece::getBoundingBox() const
{

  Point lo,hi;

  // Initialize the lo and hi points to the first element

  lo = child_[0]->getBoundingBox().lower();
  hi = child_[0]->getBoundingBox().upper();

  for( unsigned int i = 0; i < child_.size(); i++ ) {
    Box box = child_[i]->getBoundingBox();
    lo = Min(lo,box.lower());
    hi = Max(hi,box.upper());
  }

  return Box(lo,hi);
}
