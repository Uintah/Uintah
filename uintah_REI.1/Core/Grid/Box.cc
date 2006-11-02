
#include <Packages/Uintah/Core/Grid/Box.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <deque>

using namespace Uintah;
using namespace std;

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
  ostream&
  operator<<(ostream& out, const Box& b)
  {
    out << b.lower() << ".." << b.upper();
    return out;
  }
}
