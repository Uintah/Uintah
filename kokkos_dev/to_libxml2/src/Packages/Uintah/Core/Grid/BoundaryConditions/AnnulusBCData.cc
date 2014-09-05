#include <Packages/Uintah/Core/Grid/BoundaryConditions/AnnulusBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

AnnulusBCData::AnnulusBCData() 
{
  
}

AnnulusBCData::AnnulusBCData(BCData& bc)
  : d_bc(bc)
{
}

AnnulusBCData::AnnulusBCData(Point& p, double inRadius,double outRadius)
  : d_innerRadius(inRadius), d_outerRadius(outRadius), d_origin(p)
{
}

AnnulusBCData::~AnnulusBCData()
{
}

AnnulusBCData* AnnulusBCData::clone()
{
  return scinew AnnulusBCData(*this);
}

void AnnulusBCData::addBCData(BCData& bc) 
{
  d_bc = bc;
}


void AnnulusBCData::addBC(BoundCondBase* bc) 
{
  d_bc.setBCValues(bc);
}

void AnnulusBCData::getBCData(BCData& bc) const 
{
  bc = d_bc;
}

bool AnnulusBCData::inside(const Point &p) const 
{
  Vector diff = p - d_origin;

  bool inside_outer = false;
  bool outside_inner = false;

  if (diff.length() > d_outerRadius)
    inside_outer = false;
  else
    inside_outer =  true;

  if (diff.length() > d_innerRadius)
    outside_inner = true;
  else
    outside_inner = false;

  return (inside_outer && outside_inner);
  
}

void AnnulusBCData::print()
{
  cout << "Geometry type = " << typeid(this).name() << endl;
  d_bc.print();
}

void AnnulusBCData::determineIteratorLimits(Patch::FaceType face, 
					   const Patch* patch, 
					   vector<Point>& test_pts)
{
#if 0
  cout << "Annulus determineIteratorLimits()" << endl;
#endif
  BCGeomBase::determineIteratorLimits(face,patch,test_pts);
}

void AnnulusBCData::determineSFLimits(Patch::FaceType face, const Patch* patch)
{
#if 0
  cout << "Annulus determineSFLimits()" << endl;
#endif
  BCGeomBase::determineSFLimits(face,patch);
}
