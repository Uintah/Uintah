#include <Packages/Uintah/Core/Grid/BoundaryConditions/SideBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

SideBCData::SideBCData() 
{
  
}


SideBCData::SideBCData(BCData& bc) 
  : d_bc(bc)
{
}

SideBCData::~SideBCData()
{
}

SideBCData* SideBCData::clone()
{
  return scinew SideBCData(*this);

}
void SideBCData::addBCData(BCData& bc)
{
  d_bc = bc;
}

void SideBCData::addBC(BoundCondBase* bc)
{
  d_bc.setBCValues(bc);
}


void SideBCData::getBCData(BCData& bc) const
{
  bc = d_bc;
}

bool SideBCData::inside(const Point &p) const 
{
  return true;
}

void SideBCData::print()
{
  cout << "Geometry type = " << typeid(this).name() << endl;
  d_bc.print();
}

void SideBCData::determineIteratorLimits(Patch::FaceType face, 
					 const Patch* patch, 
					 vector<Point>& test_pts)
{
#if 0
  cout << "SideBC determineIteratorLimits()" << endl;
#endif
  BCGeomBase::determineIteratorLimits(face,patch,test_pts);
}

void SideBCData::determineSFLimits(Patch::FaceType face, const Patch* patch)
{
#if 0
  cout << "SideBC determineSFLimits()" << endl;
#endif
  BCGeomBase::determineSFLimits(face,patch);
}
