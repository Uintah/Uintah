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


#include <Core/Grid/BoundaryConditions/DifferenceBCData.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Grid/Variables/DifferenceIterator.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <set>
#include <iostream>
#include <algorithm>

using std::endl;

using namespace SCIRun;
using namespace Uintah;

// export SCI_DEBUG="BC_dbg:+"
static DebugStream BC_dbg("BC_dbg",false);

DifferenceBCData::DifferenceBCData(BCGeomBase* p1,BCGeomBase* p2)
  : BCGeomBase(), left(p1->clone()), right(p2->clone())
{
}


DifferenceBCData::DifferenceBCData(const DifferenceBCData& rhs): BCGeomBase(rhs)
{
  left=rhs.left->clone();
  right=rhs.right->clone();
}



DifferenceBCData& DifferenceBCData::operator=(const DifferenceBCData& rhs)
{
  BCGeomBase::operator=(rhs);

  if (this == &rhs)
    return *this;

  // Delete the lhs
  delete right;
  delete left;

  // Copy the rhs to the lhs

  left = rhs.left->clone();
  right = rhs.right->clone();

  return *this;
}

DifferenceBCData::~DifferenceBCData()
{
  delete left;
  delete right;
}


bool DifferenceBCData::operator==(const BCGeomBase& rhs) const
{
  const DifferenceBCData* p_rhs = 
    dynamic_cast<const DifferenceBCData*>(&rhs);

  if (p_rhs == NULL)
    return false;
  else
    return (this->left == p_rhs->left) && (this->right == p_rhs->right);

}

DifferenceBCData* DifferenceBCData::clone()
{
  return scinew DifferenceBCData(*this);
}

void DifferenceBCData::addBCData(BCData& bc)
{

}


void DifferenceBCData::addBC(BoundCondBase* bc)
{

}

void DifferenceBCData::getBCData(BCData& bc) const
{
  left->getBCData(bc);
}

bool DifferenceBCData::inside(const Point &p) const 
{
  return (left->inside(p) && !right->inside(p));
}

void DifferenceBCData::print()
{
#if 1
  BC_dbg << "Difference Geometry type = " << typeid(this).name() << endl;
  BC_dbg << "Left" << endl;
#endif
  left->print();
#if 1
  BC_dbg << "Right" << endl;
#endif
  right->print();
}

void DifferenceBCData::determineIteratorLimits(Patch::FaceType face,
                                               const Patch* patch,
                                               vector<Point>& test_pts)
{

#if 0
  cout << "DifferenceBC determineIteratorLimits() " << patch->getFaceName(face)<< endl;
#endif


  left->determineIteratorLimits(face,patch,test_pts);
  right->determineIteratorLimits(face,patch,test_pts);

  Iterator left_cell,left_node,right_cell,right_node;
  left->getCellFaceIterator(left_cell);
  left->getNodeFaceIterator(left_node);
  right->getCellFaceIterator(right_cell);
  right->getNodeFaceIterator(right_node);

  d_cells = DifferenceIterator(left_cell,right_cell);
  d_nodes = DifferenceIterator(left_node,right_node);


#if 0
#if 0
  cout << "DifferenceBC determineIteratorLimits()" << endl;
  cout << "Doing left determineIteratorLimits()" << endl;
#endif
  left->determineIteratorLimits(face,patch,test_pts);
#if 0
  cout << "Doing right determineIteratorLimits()" << endl;
#endif
  right->determineIteratorLimits(face,patch,test_pts);

#if 0
  cout << "Size of boundary = " << boundary.size() << endl;
  cout << "Size of nboundary = " << nboundary.size() << endl;
#endif

  // Need to do the set difference operations for the left and right to get
  // the boundary and nboundary iterators.
  vector<IntVector> diff_boundary,   diff_nboundary;
  vector<IntVector> *left_boundary,  *right_boundary;
  vector<IntVector> *left_nboundary, *right_nboundary;

  left->getBoundaryIterator(left_boundary);
  left->getNBoundaryIterator(left_nboundary);

  right->getBoundaryIterator(right_boundary);
  right->getNBoundaryIterator(right_nboundary);

#if 0
  cout << "Size of left_boundary = " << left_boundary->size() << endl;
  cout << "Size of left_nboundary = " << left_nboundary->size() << endl;
  cout << "Size of right_boundary = " << right_boundary->size() << endl;
  cout << "Size of right_nboundary = " << right_nboundary->size() << endl;
#endif
  
  for (vector<IntVector>::const_iterator it = left_boundary->begin();
       it != left_boundary->end(); ++it) {
    vector<IntVector>::const_iterator result = find(right_boundary->begin(),
                                                    right_boundary->end(),*it);
    if (result == right_boundary->end())
      diff_boundary.push_back(*it);
  }

  for (vector<IntVector>::const_iterator it = left_nboundary->begin();
       it != left_nboundary->end(); ++it) {
    vector<IntVector>::const_iterator result = find(right_nboundary->begin(),
                                                    right_nboundary->end(),*it);
    if (result == right_nboundary->end())
      diff_nboundary.push_back(*it);
  }

  setBoundaryIterator(diff_boundary);
  setNBoundaryIterator(diff_nboundary);

#if 0
  cout << "Size of boundary = " << boundary->size() << endl;
  cout << "Size of nboundary = " << nboundary->size() << endl;
#endif
  
#endif
}


