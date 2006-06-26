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

void BCGeomBase::getBoundaryIterator(vector<IntVector>& b) const
{
  b = boundary;
}

void BCGeomBase::getNBoundaryIterator(vector<IntVector>& b) const
{
  b = nboundary;
}

void BCGeomBase::getSFCXIterator(vector<IntVector>& i) const
{
  i = sfcx;
}

void BCGeomBase::getSFCYIterator(vector<IntVector>& i) const
{
  i = sfcy;
}

void BCGeomBase::getSFCZIterator(vector<IntVector>& i) const
{
  i = sfcz;
}

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

  determineSFLimits(face,patch);
}

void BCGeomBase::determineSFLimits(Patch::FaceType face, const Patch* patch)
{
#if 0
  cout << "BCGeomBase determineSFLimits()" << endl;
  cout << "Face = " << face << endl;
#endif
  set<int> same_x, same_y, same_z;
  for (vector<IntVector>::const_iterator it = boundary.begin(); 
       it != boundary.end(); ++it) {
#if 0
    cout << "boundary = " << *it << endl;
#endif
    same_x.insert((*it).x());
    same_y.insert((*it).y());
    same_z.insert((*it).z());
  }

  // Look at the extents for the components of the boundary iterator that 
  // are orthogonal to the face, i.e. x face, look at the extents for 
  // y and z.  We add 1 in each direction to come up with the extra iterators
  // that need to be added to the boundary.  For the component that is parallel
  // to the face, we use the iterators for the boundary adjusting the index,
  // so that it will be the "inside" face of the boundary cells.

  vector<IntVector> x_iterator;
  vector<IntVector> y_iterator;
  vector<IntVector> z_iterator;
  vector<IntVector> sfx,sfy,sfz;

  if (face == Patch::xplus || face == Patch::xminus) {
    for (set<int>::const_iterator y = same_y.begin(); y != same_y.end(); ++y) {
      vector<IntVector> same_y_element;
      for (vector<IntVector>::const_iterator it = boundary.begin();
	   it != boundary.end(); ++it) {
	if (*y == (*it).y())
	  same_y_element.push_back(*it);
      }
      // Find the biggest element and add (1,0,0)
      sort(same_y_element.begin(),same_y_element.end(),ltiv_z());
      z_iterator.push_back(same_y_element.back()+IntVector(0,0,1));
    }
    // add the boundary elements to the z_iterator
    copy(boundary.begin(),boundary.end(),back_inserter(z_iterator));
    
    for (set<int>::const_iterator z = same_z.begin(); z != same_z.end(); ++z) {
      vector<IntVector> same_z_element;
      for (vector<IntVector>::const_iterator it = boundary.begin();
	   it != boundary.end(); ++it) {
	if (*z == (*it).z())
	  same_z_element.push_back(*it);
      }
      // Find the biggest element and add (0,1,0)
      sort(same_z_element.begin(),same_z_element.end(),ltiv_y());
      y_iterator.push_back(same_z_element.back()+IntVector(0,1,0));
    }
    
    // add the boundary element to the y_iterator
    copy(boundary.begin(),boundary.end(),back_inserter(y_iterator));
    
  }

  if (face == Patch::yplus || face == Patch::yminus) {
    for (set<int>::const_iterator z = same_z.begin(); z != same_z.end(); ++z) {
      vector<IntVector> same_z_element;
      for (vector<IntVector>::const_iterator it = boundary.begin();
	   it != boundary.end(); ++it) {
	if (*z == (*it).z())
	  same_z_element.push_back(*it);
      }
      // Find the biggest element and add (1,0,0)
      sort(same_z_element.begin(),same_z_element.end(),ltiv_x());
      x_iterator.push_back(same_z_element.back()+IntVector(1,0,0));
    }
    // add the boundary elements to the x_iterator
    copy(boundary.begin(),boundary.end(),back_inserter(x_iterator));
    
    for (set<int>::const_iterator x = same_x.begin(); x != same_x.end(); ++x) {
      vector<IntVector> same_x_element;
      for (vector<IntVector>::const_iterator it = boundary.begin();
	   it != boundary.end(); ++it) {
	if (*x == (*it).x())
	  same_x_element.push_back(*it);
      }
      // Find the biggest element and add (0,1,0)
      sort(same_x_element.begin(),same_x_element.end(),ltiv_z());
      z_iterator.push_back(same_x_element.back()+IntVector(0,0,1));
    }
    // add the boundary element to the y_iterator
    copy(boundary.begin(),boundary.end(),back_inserter(z_iterator));
        
  }
  if (face == Patch::zplus || face == Patch::zminus) {
    for (set<int>::const_iterator y = same_y.begin(); y != same_y.end(); ++y) {
      vector<IntVector> same_y_element;
      for (vector<IntVector>::const_iterator it = boundary.begin();
	   it != boundary.end(); ++it) {
	if (*y == (*it).y())
	  same_y_element.push_back(*it);
      }
      // Find the biggest element and add (1,0,0)
      sort(same_y_element.begin(),same_y_element.end(),ltiv_x());
      x_iterator.push_back(same_y_element.back()+IntVector(1,0,0));
    }
    // add the boundary elements to the x_iterator
    copy(boundary.begin(),boundary.end(),back_inserter(x_iterator));
    
    for (set<int>::const_iterator x = same_x.begin(); x != same_x.end(); ++x) {
      vector<IntVector> same_x_element;
      for (vector<IntVector>::const_iterator it = boundary.begin();
	   it != boundary.end(); ++it) {
	if (*x == (*it).x())
	  same_x_element.push_back(*it);
      }
      // Find the biggest element and add (0,1,0)
      sort(same_x_element.begin(),same_x_element.end(),ltiv_y());
      y_iterator.push_back(same_x_element.back()+IntVector(0,1,0));
    }
    // add the boundary element to the y_iterator
    copy(boundary.begin(),boundary.end(),back_inserter(y_iterator));
    
  }

  switch(face) {
  case Patch::xminus:
    {
      transform(boundary.begin(),boundary.end(),back_inserter(sfx),
		bind2nd(plus<IntVector>(),IntVector(1,0,0)));
      transform(y_iterator.begin(),y_iterator.end(),back_inserter(sfy),
		bind2nd(plus<IntVector>(),IntVector(1,0,0)));
      transform(z_iterator.begin(),z_iterator.end(),back_inserter(sfz),
		bind2nd(plus<IntVector>(),IntVector(1,0,0)));
      break;
    }
  case Patch::xplus:
    {
      transform(boundary.begin(),boundary.end(),back_inserter(sfx),
		bind2nd(plus<IntVector>(),IntVector(-1,0,0)));
      transform(y_iterator.begin(),y_iterator.end(),back_inserter(sfy),
		bind2nd(plus<IntVector>(),IntVector(-1,0,0)));
      transform(z_iterator.begin(),z_iterator.end(),back_inserter(sfz),
		bind2nd(plus<IntVector>(),IntVector(-1,0,0)));
      break;
    }
  case Patch::yminus:
    {
      transform(boundary.begin(),boundary.end(),back_inserter(sfy),
		bind2nd(plus<IntVector>(),IntVector(0,1,0)));
      transform(x_iterator.begin(),x_iterator.end(),back_inserter(sfx),
		bind2nd(plus<IntVector>(),IntVector(0,1,0)));
      transform(z_iterator.begin(),z_iterator.end(),back_inserter(sfz),
		bind2nd(plus<IntVector>(),IntVector(0,1,0)));
      break;
    }
  case Patch::yplus:
    {
      transform(boundary.begin(),boundary.end(),back_inserter(sfy),
		bind2nd(plus<IntVector>(),IntVector(0,-1,0)));
      transform(x_iterator.begin(),x_iterator.end(),back_inserter(sfx),
		bind2nd(plus<IntVector>(),IntVector(0,-1,0)));
      transform(z_iterator.begin(),z_iterator.end(),back_inserter(sfz),
		bind2nd(plus<IntVector>(),IntVector(0,-1,0)));
      break;
    }
  case Patch::zminus:
    {
      transform(boundary.begin(),boundary.end(),back_inserter(sfz),
		bind2nd(plus<IntVector>(),IntVector(0,0,1)));
      transform(x_iterator.begin(),x_iterator.end(),back_inserter(sfx),
		bind2nd(plus<IntVector>(),IntVector(0,0,1)));
      transform(y_iterator.begin(),y_iterator.end(),back_inserter(sfy),
		bind2nd(plus<IntVector>(),IntVector(0,0,1)));
      break;
    }
  case Patch::zplus:
    {
      transform(boundary.begin(),boundary.end(),back_inserter(sfz),
		bind2nd(plus<IntVector>(),IntVector(0,0,-1)));
      transform(x_iterator.begin(),x_iterator.end(),back_inserter(sfx),
		bind2nd(plus<IntVector>(),IntVector(0,0,-1)));
      transform(y_iterator.begin(),y_iterator.end(),back_inserter(sfy),
		bind2nd(plus<IntVector>(),IntVector(0,0,-1)));
      break;
    }
  default:
    break;
  }
  setSFCXIterator(sfx);
  setSFCYIterator(sfy);
  setSFCZIterator(sfz);
#if 0
  cout << "Size of sfcx = " << sfcx.size() << endl;
  for (vector<IntVector>::const_iterator it = sfcx.begin(); it != sfcx.end();
       ++it) 
    cout << "sfcx = " << *it << endl;
  cout << "Size of sfcy = " << sfcy.size() << endl;
  for (vector<IntVector>::const_iterator it = sfcy.begin(); it != sfcy.end();
       ++it) 
    cout << "sfcy = " << *it << endl;
  cout << "Size of sfcz = " << sfcz.size() << endl;
  for (vector<IntVector>::const_iterator it = sfcz.begin(); it != sfcz.end();
       ++it) 
    cout << "sfcz = " << *it << endl;
#endif

}

void BCGeomBase::printLimits() const
{
  cout << endl;
  for (vector<IntVector>::const_iterator it = boundary.begin(); 
       it != boundary.end(); ++it)
    cout << "boundary = " << *it << endl;
  cout << endl;

}
