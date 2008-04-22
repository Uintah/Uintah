#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <set>
#include <vector>
#include <functional> // for bind2nd
#include <sgi_stl_warnings_on.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

BCGeomBase::BCGeomBase() 
{
}

BCGeomBase::~BCGeomBase()
{
}

void BCGeomBase::setBoundaryIterator(vector<IntVector>& b)
{
  boundary = b;
}

void BCGeomBase::setNBoundaryIterator(vector<IntVector>& b)
{
  nboundary = b;
}

void BCGeomBase::setBoundaryIterator(vector<IntVector>::iterator b,
                                     vector<IntVector>::iterator e)
{
  b_b = b;
  b_e = e;
}

void BCGeomBase::setNBoundaryIterator(vector<IntVector>::iterator b,
                                      vector<IntVector>::iterator e)
{
  nb_b = b;
  nb_e = e;
}


#if 0
void BCGeomBase::setSFCXIterator(vector<IntVector>& i)
{
  sfcx = i;
}

void BCGeomBase::setSFCYIterator(vector<IntVector>& i)
{
  sfcy = i;
}

void BCGeomBase::setSFCZIterator(vector<IntVector>& i)
{
  sfcz = i;
}
#endif
void BCGeomBase::getBoundaryIterator(vector<IntVector>*& b_ptr) 
{
  b_ptr = &boundary;
}

void BCGeomBase::getNBoundaryIterator(vector<IntVector>*& b_ptr)
{
  b_ptr = &nboundary;
}
#if 0
void BCGeomBase::getSFCXIterator(vector<IntVector>*& i_ptr)
{
  i_ptr = &sfcx;
}

void BCGeomBase::getSFCYIterator(vector<IntVector>*& i_ptr)
{
  i_ptr = &sfcy;
}

void BCGeomBase::getSFCZIterator(vector<IntVector>*& i_ptr)
{
  i_ptr = &sfcz;
}

#endif
void BCGeomBase::determineIteratorLimits(Patch::FaceType face, 
					 const Patch* patch, 
					 vector<Point>& test_pts)
{
#if 0
  cout << "BCGeomBase determineIteratorLimits()" << endl;
#endif
  IntVector l,h;
  patch->getFaceCells(face,0,l,h);

  vector<IntVector> b,nb;

  vector<Point>::const_iterator pts = test_pts.begin();
  for (CellIterator bound(l,h); !bound.done(); bound++,pts++) 
    if (inside(*pts))
      b.push_back(*bound);
  
  IntVector ln,hn;
  patch->getFaceNodes(face,0,ln,hn);
  for (NodeIterator bound(ln,hn);!bound.done();bound++) {
    Point p = patch->getLevel()->getNodePosition(*bound);
    if (inside(p)) 
      nb.push_back(*bound);
  }

  setBoundaryIterator(b);
  setNBoundaryIterator(nb);

}

void BCGeomBase::printLimits() const
{
  cout << endl;
  for (vector<IntVector>::const_iterator it = boundary.begin(); 
       it != boundary.end(); ++it)
    cout << "boundary = " << *it << endl;
  cout << endl;

}
