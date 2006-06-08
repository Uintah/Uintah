#include <Packages/Uintah/Core/Grid/BoundaryConditions/CircleBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

CircleBCData::CircleBCData() 
{
  
}

CircleBCData::CircleBCData(BCData& bc)
  : d_bc(bc)
{
}

CircleBCData::CircleBCData(Point& p, double radius)
  : d_radius(radius), d_origin(p)
{
}

CircleBCData::~CircleBCData()
{
}

CircleBCData* CircleBCData::clone()
{
  return scinew CircleBCData(*this);
}

void CircleBCData::addBCData(BCData& bc) 
{
  d_bc = bc;
}


void CircleBCData::addBC(BoundCondBase* bc) 
{
  d_bc.setBCValues(bc);
}

void CircleBCData::getBCData(BCData& bc) const 
{
  bc = d_bc;
}

bool CircleBCData::inside(const Point &p) const 
{
  Vector diff = p - d_origin;
  if (diff.length() > d_radius)
    return false;
  else
    return true;
}

void CircleBCData::print()
{
  cout << "Geometry type = " << typeid(this).name() << endl;
  d_bc.print();
}

void CircleBCData::determineIteratorLimits(Patch::FaceType face, 
					   const Patch* patch, 
					   vector<Point>& test_pts)
{
#if 0
  cout << "Circle determineIteratorLimits()" << endl;
#endif
  BCGeomBase::determineIteratorLimits(face,patch,test_pts);
}

void CircleBCData::determineSFLimits(Patch::FaceType face, const Patch* patch)
{
#if 0
  cout << "Circle determineSFLimits()" << endl;
#endif
  BCGeomBase::determineSFLimits(face,patch);
}
