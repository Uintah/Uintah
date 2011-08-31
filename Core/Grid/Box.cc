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



#include <Core/Grid/Box.h>
#include <iostream>

#include <deque>

using namespace Uintah;


// A 'Box' is specified by two (3D) points.  However, all components (x,y,z) of the fist point (p1) must be less
// than the corresponding component of the 2nd point (p2), or the Box is considered 'degenerate()'.  If you know
// that your box is valid, this function will run through each component and make sure that the lesser value is
// stored in p1 and the greater value in p2.
void
Box::fixBoundingBox()
{
  for( int index = 0; index < 3; index++ ) {
    if( d_upper( index ) < d_lower( index ) ) {
      double temp = d_upper( index );
      d_upper( index ) = d_lower( index );
      d_lower( index ) = temp;
    }
  }
}

bool
Box::overlaps(const Box& otherbox, double epsilon) const
{
  if(d_lower.x()+epsilon > otherbox.d_upper.x() || d_upper.x() < otherbox.d_lower.x()+epsilon) {
    return false;
  }
  if(d_lower.y()+epsilon > otherbox.d_upper.y() || d_upper.y() < otherbox.d_lower.y()+epsilon) {
    return false;
  }
  if(d_lower.z()+epsilon > otherbox.d_upper.z() || d_upper.z() < otherbox.d_lower.z()+epsilon) {
    return false;
  }
  return true;
}

//static 
deque<Box> Box::difference(const Box& b1, const Box& b2)
{
  deque<Box> set1, set2;
  set1.push_back(b1);
  set2.push_back(b2);
  return difference(set1, set2);
}

//static 
deque<Box> Box::difference(deque<Box>& boxSet1, deque<Box>& boxSet2)
{
  // use 2 deques, as inserting into a deque invalidates the iterators
  deque<Box> searchSet(boxSet1.begin(), boxSet1.end());
  deque<Box> remainingBoxes;
  // loop over set2, as remainingBoxes will more than likely change
  for (unsigned i = 0; i < boxSet2.size(); i++) {
    for (deque<Box>::iterator iter = searchSet.begin(); iter != searchSet.end(); iter++) {
      Box b1 = *iter;
      Box b2 = boxSet2[i];
      if (b1.overlaps(b2)) {
        // divide the difference space into up to 6 boxes, 2 in each dimension.
        // each pass, reduce the amount of space to take up.
        // Add new boxes to the front so we don't loop over them again for this box.
        Box intersection = b1.intersect(b2);
        Box leftoverSpace = b1;
        for (int dim = 0; dim < 3; dim++) {
          if (b1.d_lower(dim) < intersection.d_lower(dim)) {
            Box tmp = leftoverSpace;
            tmp.d_lower(dim) = b1.d_lower(dim);
            tmp.d_upper(dim) = intersection.d_lower(dim);
            remainingBoxes.push_back(tmp);
          }
          if (b1.d_upper(dim) > intersection.d_upper(dim)) {
            Box tmp = leftoverSpace;
            tmp.d_lower(dim) = intersection.d_upper(dim);
            tmp.d_upper(dim) = b1.d_upper(dim);
            remainingBoxes.push_back(tmp);              
          }
          leftoverSpace.d_lower(dim) = intersection.d_lower(dim);
          leftoverSpace.d_upper(dim) = intersection.d_upper(dim);
        }
      } 
      else {
        remainingBoxes.push_back(b1);
      }
    }
    if (i+1 < boxSet2.size()) {
      searchSet = remainingBoxes;
      remainingBoxes.clear();
    }
  }
  return remainingBoxes;
}

namespace Uintah {
  std::ostream&
  operator<<(std::ostream& out, const Box& b)
  {
    out << b.lower() << ".." << b.upper();
    return out;
  }
}
