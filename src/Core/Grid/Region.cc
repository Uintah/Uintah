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



#include <Core/Grid/Region.h>
#include <iostream>

#include <vector>

using namespace std;
using namespace Uintah;


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
vector<Region> Region::difference(const Region& b1, const Region& b2)
{
  vector<Region> set1, set2;
  set1.push_back(b1);
  set2.push_back(b2);
  return difference(set1, set2);
}

//static 
vector<Region> Region::difference(vector<Region>& region1, vector<Region>& region2)
{
  vector<Region> searchSet(region1);
  vector<Region> remainingRegions;

  // loop over set2, as remainingRegiones will more than likely change
  for (unsigned i = 0; i < region2.size(); i++) {
    for (vector<Region>::iterator iter = searchSet.begin(); iter != searchSet.end(); iter++) {
      Region b1 = *iter;
      Region b2 = region2[i];
      if (b1.overlaps(b2)) {
        // divide the difference space into up to 6 regions, 2 in each dimension.
        // each pass, reduce the amount of space to take up.
        Region intersection = b1.intersect(b2);
        Region leftoverSpace = b1;
        for (int dim = 0; dim < 3; dim++) {
          if (b1.d_lowIndex(dim) < intersection.d_lowIndex(dim)) {
            Region tmp = leftoverSpace;
            tmp.d_lowIndex(dim) = b1.d_lowIndex(dim);
            tmp.d_highIndex(dim) = intersection.d_lowIndex(dim);
            if(tmp.getVolume()>0)
              remainingRegions.push_back(tmp);
          }
          if (b1.d_highIndex(dim) > intersection.d_highIndex(dim)) {
            Region tmp = leftoverSpace;
            tmp.d_lowIndex(dim) = intersection.d_highIndex(dim);
            tmp.d_highIndex(dim) = b1.d_highIndex(dim);
            if(tmp.getVolume()>0)
              remainingRegions.push_back(tmp);              
          }
          leftoverSpace.d_lowIndex(dim) = intersection.d_lowIndex(dim);
          leftoverSpace.d_highIndex(dim) = intersection.d_highIndex(dim);
        }
      } 
      else {
        remainingRegions.push_back(b1);
      }
    }
    
    //update the search set to any remaining regions
    searchSet = remainingRegions;
    remainingRegions.clear();
  }
  return searchSet;
  //return remainingRegions;
}

