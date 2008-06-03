#include <Packages/Uintah/Core/Grid/BoundaryConditions/SideBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
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

#if 0
  BCGeomBase::determineIteratorLimits(face,patch,test_pts);
#else
  IntVector l,h;
  patch->getFaceCells(face,0,l,h);
  vector<IntVector> b,nb;


  for (CellIterator bound(l,h); !bound.done(); bound++) 
    b.push_back(*bound);
  
  IntVector ln,hn;
  patch->getFaceNodes(face,0,ln,hn);
  for (NodeIterator bound(ln,hn);!bound.done();bound++) {
    nb.push_back(*bound);
  }


#if 1
  setBoundaryIterator(b);
  setNBoundaryIterator(nb);
#endif
  setBoundaryIterator(b.begin(),b.end());
  setNBoundaryIterator(nb.begin(),nb.end());


#if 0
  determineSFLimits(face,patch);
#endif

#endif  

  
}

#if 0
void SideBCData::determineSFLimits(Patch::FaceType face, const Patch* patch)
{
#if 0
  cout << "SideBC determineSFLimits()" << endl;
  BCGeomBase::determineSFLimits(face,patch);
#endif
  return;
}
#endif
