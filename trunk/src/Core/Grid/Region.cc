
#include <Packages/Uintah/Core/Grid/Region.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <deque>

using namespace Uintah;
using namespace std;

//This code was pulled from Box.cc

bool
Region::overlaps(const Region& otherregion) const
{
  if(d_lowIndex.x() > otherregion.d_highIndex.x() || d_highIndex.x() < otherregion.d_lowIndex.x()) {
    return false;
  }
  if(d_lowIndex.y() > otherregion.d_highIndex.y() || d_highIndex.y() < otherregion.d_lowIndex.y()) {
    return false;
  }
  if(d_lowIndex.z() > otherregion.d_highIndex.z() || d_highIndex.z() < otherregion.d_lowIndex.z()) {
    return false;
  }
  return true;
}

//static 
deque<Region> Region::difference(const Region& b1, const Region& b2)
{
  deque<Region> set1, set2;
  set1.push_back(b1);
  set2.push_back(b2);
  return difference(set1, set2);
}

//static 
deque<Region> Region::difference(deque<Region>& region1, deque<Region>& region2)
{
  // use 2 deques, as inserting into a deque invalidates the iterators
  deque<Region> searchSet(region1.begin(), region1.end());
  deque<Region> remainingRegiones;
  // loop over set2, as remainingRegiones will more than likely change
  for (unsigned i = 0; i < region2.size(); i++) {
    for (deque<Region>::iterator iter = searchSet.begin(); iter != searchSet.end(); iter++) {
      Region b1 = *iter;
      Region b2 = region2[i];
      if (b1.overlaps(b2)) {
        // divide the difference space into up to 6 regions, 2 in each dimension.
        // each pass, reduce the amount of space to take up.
        // Add new regions to the front so we don't loop over them again for this region.
        Region intersection = b1.intersect(b2);
        Region leftoverSpace = b1;
        for (int dim = 0; dim < 3; dim++) {
          if (b1.d_lowIndex(dim) < intersection.d_lowIndex(dim)) {
            Region tmp = leftoverSpace;
            tmp.d_lowIndex(dim) = b1.d_lowIndex(dim);
            tmp.d_highIndex(dim) = intersection.d_lowIndex(dim);
            remainingRegiones.push_back(tmp);
          }
          if (b1.d_highIndex(dim) > intersection.d_highIndex(dim)) {
            Region tmp = leftoverSpace;
            tmp.d_lowIndex(dim) = intersection.d_highIndex(dim);
            tmp.d_highIndex(dim) = b1.d_highIndex(dim);
            remainingRegiones.push_back(tmp);              
          }
          leftoverSpace.d_lowIndex(dim) = intersection.d_lowIndex(dim);
          leftoverSpace.d_highIndex(dim) = intersection.d_highIndex(dim);
        }
      } 
      else {
        remainingRegiones.push_back(b1);
      }
    }
    if (i+1 < region2.size()) {
      searchSet = remainingRegiones;
      remainingRegiones.clear();
    }
  }
  return remainingRegiones;
}

