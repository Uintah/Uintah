#include <Packages/Uintah/Core/Grid/BoundaryConditions/DifferenceBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Malloc/Allocator.h>
#include <set>
#include <iostream>
#include <algorithm>
using namespace SCIRun;
using namespace Uintah;
using namespace std;

DifferenceBCData::DifferenceBCData(BCGeomBase* p1,BCGeomBase* p2)
  : left(p1->clone()), right(p2->clone())
{
}

DifferenceBCData::DifferenceBCData(const DifferenceBCData& rhs)
{
  left=rhs.left->clone();
  right=rhs.right->clone();

  boundary=rhs.boundary;
  nboundary=rhs.nboundary;
#if 0
  sfcx=rhs.sfcx;
  sfcy=rhs.sfcy;
  sfcz=rhs.sfcz;
#endif
}

DifferenceBCData& DifferenceBCData::operator=(const DifferenceBCData& rhs)
{
  if (this == &rhs)
    return *this;

  // Delete the lhs
  delete right;
  delete left;

  // Copy the rhs to the lhs

  left = rhs.left->clone();
  right = rhs.right->clone();

  boundary = rhs.boundary;
  nboundary = rhs.nboundary;
#if 0
  sfcx=rhs.sfcx;
  sfcy=rhs.sfcy;
  sfcz=rhs.sfcz;
#endif
  return *this;
}

DifferenceBCData::~DifferenceBCData()
{
  delete left;
  delete right;
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
  cout << "Geometry type = " << typeid(this).name() << endl;
  left->print();
  right->print();
}

void DifferenceBCData::determineIteratorLimits(Patch::FaceType face,
					       const Patch* patch,
					       vector<Point>& test_pts)
{
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
  
}


