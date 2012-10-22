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

#include <TauProfilerForSCIRun.h>

#include <Core/Grid/PatchRangeTree.h>
#include <list>

using namespace std;
using namespace Uintah;
using namespace SCIRun;


PatchRangeTree::PatchRangeTree(const std::vector<Patch*>& patches)
  :  d_maxPatchDimensions(0, 0, 0),
     d_patchPoints(new PatchPoint[patches.size()]),
     d_numPatches((int)patches.size())
{
  TAU_PROFILE("PatchRangeTree::PatchRangeTree-a", " ", TAU_USER);
  list<PatchPoint*> pointList;
  IntVector dimensions;
  
  for (int i = 0; i < (int)patches.size(); i++) {
    d_patchPoints[i].setPatch(patches[i]);
    pointList.push_back(&d_patchPoints[i]);
    
    dimensions =
      patches[i]->getExtraNodeHighIndex() - patches[i]->getExtraNodeLowIndex();

    for (int j = 0; j < 3; j++) {
      if (dimensions[j] > d_maxPatchDimensions[j]) {
        d_maxPatchDimensions[j] = dimensions[j];
      }
    }
  }

  d_rangeTree = scinew RangeTree<PatchPoint, int>(pointList, 3 /*dimensions*/);
}

PatchRangeTree::PatchRangeTree(const std::vector<const Patch*>& patches)
  :  d_maxPatchDimensions(0, 0, 0),
     d_patchPoints(new PatchPoint[patches.size()]),
     d_numPatches((int)patches.size())
{
  TAU_PROFILE("PatchRangeTree::PatchRangeTree-b", " ", TAU_USER);
  list<PatchPoint*> pointList;
  IntVector dimensions;
  
  for (int i = 0; i < (int)patches.size(); i++) {
    d_patchPoints[i].setPatch(patches[i]);
    pointList.push_back(&d_patchPoints[i]);
    
    dimensions =
      patches[i]->getExtraNodeHighIndex() - patches[i]->getExtraNodeLowIndex();

    for (int j = 0; j < 3; j++) {
      if (dimensions[j] > d_maxPatchDimensions[j]) {
        d_maxPatchDimensions[j] = dimensions[j];
      }
    }
  }

  d_rangeTree = scinew RangeTree<PatchPoint, int>(pointList, 3 /*dimensions*/);
}

PatchRangeTree::~PatchRangeTree()
{
  delete d_rangeTree;
  delete[] d_patchPoints;
}

void PatchRangeTree::query(const IntVector& low, const IntVector& high,
                           Patch::selectType& foundPatches)
{
  list<PatchPoint*> foundPoints;

  // Note: factor of 2 is to make calculations simple and not
  // require rounding, but think of this as doing a query on
  // the patch centers and think of these query values as halved.
  IntVector centerLowTimes2 =
    low * IntVector(2, 2, 2) - d_maxPatchDimensions;
  IntVector centerHighTimes2 =
    high * IntVector(2, 2, 2) + d_maxPatchDimensions;

  PatchPoint lowPatchPoint(centerLowTimes2);
  PatchPoint highPatchPoint(centerHighTimes2);

  d_rangeTree->query(lowPatchPoint, highPatchPoint, foundPoints);

  // So far we have found all of the patches that can be in the
  // range (and would be if they all had the same dimensions).  Now
  // just go through the list of the ones found and report the ones
  // that actually are in range.  (The assumption here is that most
  // of the ones found above are actually in the range -- this asumption
  // is valid iff the maximum patch dimensions are not much larger than
  // the average patch dimensions).

  //foundPatches.reserve(foundPatches.size() + foundPoints.size());
  for (list<PatchPoint*>::iterator it = foundPoints.begin();
       it != foundPoints.end(); it++) {    
    const Patch* patch = (*it)->getPatch();
    IntVector l=Max(patch->getCellLowIndex(), low);
    IntVector u=Min(patch->getCellHighIndex(), high);
    if (u.x() > l.x() && u.y() > l.y() && u.z() > l.z())
      foundPatches.push_back(patch);
  }
}
