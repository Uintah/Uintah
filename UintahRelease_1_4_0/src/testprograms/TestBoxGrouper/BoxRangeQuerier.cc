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


#include <testprograms/TestBoxGrouper/Box.h>
#include <testprograms/TestBoxGrouper/BoxRangeQuerier.h>
#include <list>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

BoxRangeQuerier::~BoxRangeQuerier()
{
  delete d_rangeTree;
}

void BoxRangeQuerier::query(const IntVector& low, const IntVector& high,
			    list<const Box*>& foundBoxes)
{
  list<BoxPoint*> foundPoints;

  // Note: factor of 2 is to make calculations simple and not
  // require rounding, but think of this as doing a query on
  // the box centers and think of these query values as halved.
  IntVector centerLowTimes2 =
    low * IntVector(2, 2, 2) - d_maxBoxDimensions;
  IntVector centerHighTimes2 =
    high * IntVector(2, 2, 2) + d_maxBoxDimensions;

  BoxPoint lowBoxPoint(centerLowTimes2);
  BoxPoint highBoxPoint(centerHighTimes2);

  d_rangeTree->query(lowBoxPoint, highBoxPoint, foundPoints);

  // So far we have found all of the boxes that can be in the
  // range (and would be if they all had the same dimensions).  Now
  // just go through the list of the ones found and report the ones
  // that actually are in range.  (The assumption here is that most
  // of the ones found above are actually in the range -- this asumption
  // is valid iff the maximum box dimensions are not much larger than
  // the average box dimensions).

  //foundBoxes.reserve(foundBoxes.size() + foundPoints.size());
  for (list<BoxPoint*>::iterator it = foundPoints.begin();
       it != foundPoints.end(); it++) {    
    const Box* box = (*it)->getBox();
    if (box->isInside(low, high))
      foundBoxes.push_back(box);
  }
}

void
BoxRangeQuerier::queryNeighbors(const IntVector& low, const IntVector& high,
				list<const Box*>& foundBoxes)
{
  list<BoxPoint*> foundPoints;
  for (int i = 0; i < 3; i++) {
    IntVector sideLow = low; --sideLow[i];
    IntVector sideHigh = high; sideHigh[i] = sideLow[i];
    query(sideLow, sideHigh, foundBoxes);
    sideHigh = high; ++sideHigh[i];
    sideLow = low; sideLow[i] = sideHigh[i];
    query(sideLow, sideHigh, foundBoxes);
  }
  for (list<const Box*>::iterator iter = foundBoxes.begin(); iter != foundBoxes.end(); iter++) {
    ASSERT((*iter)->isNeighboring(low, high));  
  }
}

/*
list<const Box*> BoxRangeQuerier::query(const IntVector& low, const IntVector& high)
{
  list<const Box*> results;
  for (unsigned long i = 0; i < d_boxPoints.size(); i++) {
    const Box* box = d_boxPoints[i].getBox();
    if (box->isInside(low, high))
      results.push_back(box);
  }
  return results;
}

list<const Box*> BoxRangeQuerier::queryNeighbors(const IntVector& low, const IntVector& high)
{
  list<const Box*> results;
  for (unsigned long i = 0; i < d_boxPoints.size(); i++) {
    const Box* box = d_boxPoints[i].getBox();
    if (box->isNeighboring(low, high))
      results.push_back(box);
  }
  return results;
}
*/
