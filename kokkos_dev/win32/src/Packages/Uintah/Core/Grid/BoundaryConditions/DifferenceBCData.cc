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
  sfcx=rhs.sfcx;
  sfcy=rhs.sfcy;
  sfcz=rhs.sfcz;
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
  sfcx=rhs.sfcx;
  sfcy=rhs.sfcy;
  sfcz=rhs.sfcz;
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

  vector<IntVector> left_boundary,right_boundary,diff_boundary;
  vector<IntVector> left_nboundary,right_nboundary,diff_nboundary;

  left->getBoundaryIterator(left_boundary);
  left->getNBoundaryIterator(left_nboundary);

  right->getBoundaryIterator(right_boundary);
  right->getNBoundaryIterator(right_nboundary);

#if 0
  cout << "Size of left_boundary = " << left_boundary.size() << endl;
  cout << "Size of left_nboundary = " << left_nboundary.size() << endl;
  cout << "Size of right_boundary = " << right_boundary.size() << endl;
  cout << "Size of right_nboundary = " << right_nboundary.size() << endl;
#endif
  
  for (vector<IntVector>::const_iterator it = left_boundary.begin();
       it != left_boundary.end(); ++it) {
    vector<IntVector>::const_iterator result = find(right_boundary.begin(),
						    right_boundary.end(),*it);
    if (result == right_boundary.end())
      diff_boundary.push_back(*it);
  }

  for (vector<IntVector>::const_iterator it = left_nboundary.begin();
       it != left_nboundary.end(); ++it) {
    vector<IntVector>::const_iterator result = find(right_nboundary.begin(),
						    right_nboundary.end(),*it);
    if (result == right_nboundary.end())
      diff_nboundary.push_back(*it);
  }

  setBoundaryIterator(diff_boundary);
  setNBoundaryIterator(diff_nboundary);

#if 0
  cout << "Size of boundary = " << boundary.size() << endl;
  cout << "Size of nboundary = " << nboundary.size() << endl;
#endif
  
  determineSFLimits(face,patch);

}


void DifferenceBCData::determineSFLimits(Patch::FaceType face, 
					 const Patch* patch)
{
#if 0
  cout << "DifferenceBC determineSFLimits()" << endl;
#endif
  // The side bc that is stored as the left pointer doesn't have the boundary
  // iterators.  The boundary iterators that were determined are for the
  // Difference bc.  Must determine the boundary iterators so that the 
  // determineSFLimits for the Side bc are determined correctly.

  // Left  == SideBC
  // Right == UnionBC

  vector<IntVector> left_sfcx,left_sfcy,left_sfcz,left_bound;
  
  left->getSFCXIterator(left_sfcx);
  left->getSFCYIterator(left_sfcy);
  left->getSFCZIterator(left_sfcz);

  vector<IntVector> right_sfcx,right_sfcy,right_sfcz;


  // The UnionBC has its iterator limits determined.

  right->getSFCXIterator(right_sfcx);
  right->getSFCYIterator(right_sfcy);
  right->getSFCZIterator(right_sfcz);
  // Need to take the difference of the left and right and then store it.

  vector<IntVector> sfx,sfy,sfz;

  for (vector<IntVector>::const_iterator it = left_sfcx.begin();
       it != left_sfcx.end(); ++it) {
    vector<IntVector>::const_iterator result = find(right_sfcx.begin(),
						    right_sfcx.end(),*it);
    if (result == right_sfcx.end())
      sfx.push_back(*it);
  }

  for (vector<IntVector>::const_iterator it = left_sfcy.begin();
       it != left_sfcy.end(); ++it) {
    vector<IntVector>::const_iterator result = find(right_sfcy.begin(),
						    right_sfcy.end(),*it);
    if (result == right_sfcy.end())
      sfy.push_back(*it);
  }

  for (vector<IntVector>::const_iterator it = left_sfcz.begin();
       it != left_sfcz.end(); ++it) {
    vector<IntVector>::const_iterator result = find(right_sfcz.begin(),
						    right_sfcz.end(),*it);
    if (result == right_sfcz.end())
      sfz.push_back(*it);
  }

  setSFCXIterator(sfx);
  setSFCYIterator(sfy);
  setSFCZIterator(sfz);

#if 0
  cout << "Size of left_sfcx = " << left_sfcx.size() << endl;
  for (vector<IntVector>::const_iterator it = left_sfcx.begin();
       it != left_sfcx.end(); ++it)
    cout << "left_sfcx = " << *it << endl;
  cout << "Size of left_sfcy = " << left_sfcy.size() << endl;
  for (vector<IntVector>::const_iterator it = left_sfcy.begin();
       it != left_sfcy.end(); ++it)
    cout << "left_sfcy = " << *it << endl;
  cout << "Size of left_sfcz = " << left_sfcz.size() << endl;
  for (vector<IntVector>::const_iterator it = left_sfcz.begin();
       it != left_sfcz.end(); ++it)
    cout << "left_sfcz = " << *it << endl;
  cout << "Size of right_sfcx = " << right_sfcx.size() << endl;
  for (vector<IntVector>::const_iterator it = right_sfcx.begin();
       it != right_sfcx.end(); ++it)
    cout << "right_sfcx = " << *it << endl;
  cout << "Size of right_sfcy = " << right_sfcy.size() << endl;
  for (vector<IntVector>::const_iterator it = right_sfcy.begin();
       it != right_sfcy.end(); ++it)
    cout << "right_sfcy = " << *it << endl;
  cout << "Size of right_sfcz = " << right_sfcz.size() << endl;
  for (vector<IntVector>::const_iterator it = right_sfcz.begin();
       it != right_sfcz.end(); ++it)
    cout << "right_sfcz = " << *it << endl;

  cout << "Size of sfcx = " << sfcx.size() << endl;
  for (vector<IntVector>::const_iterator it = sfcx.begin();
       it != sfcx.end(); ++it)
    cout << "sfcx = " << *it << endl;
  cout << "Size of sfcy = " << sfcy.size() << endl;
  for (vector<IntVector>::const_iterator it = sfcy.begin();
       it != sfcy.end(); ++it)
    cout << "sfcy = " << *it << endl;
  cout << "Size of sfcz = " << sfcz.size() << endl;
  for (vector<IntVector>::const_iterator it = sfcz.begin();
       it != sfcz.end(); ++it)
    cout << "sfcz = " << *it << endl;
#endif

}
