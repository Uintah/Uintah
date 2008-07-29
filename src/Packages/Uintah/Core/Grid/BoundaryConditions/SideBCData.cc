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


SideBCData::SideBCData() 
{
  d_cells=GridIterator(IntVector(0,0,0),IntVector(0,0,0));
  d_nodes=GridIterator(IntVector(0,0,0),IntVector(0,0,0));
}


SideBCData::~SideBCData()
{
}

bool SideBCData::operator==(const BCGeomBase& rhs) const
{
  const SideBCData* p_rhs = 
    dynamic_cast<const SideBCData*>(&rhs);

  if (p_rhs == NULL)
    return false;
  else
    return true;
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
  //cout << "Geometry type = " << typeid(this).name() << endl;
  d_bc.print();
}


void SideBCData::determineIteratorLimits(Patch::FaceType face, 
					 const Patch* patch, 
					 vector<Point>& test_pts)
{
#if 0
  cout << "SideBC determineIteratorLimits()" << endl;
#endif


  IntVector l,h;
  patch->getFaceCells(face,0,l,h);
#ifdef OLD
  d_cells = scinew GridIterator(l,h);
#else
  d_cells = GridIterator(l,h);
#endif

#if 0
  cout << "d_cells->begin() = " << d_cells->begin() << " d_cells->end() = " 
       << d_cells->end() << endl;
#endif


  IntVector ln,hn;
  patch->getFaceNodes(face,0,ln,hn);
#ifdef OLD
  d_nodes = scinew GridIterator(ln,hn);
#else
  d_nodes = GridIterator(ln,hn);
#endif


#if 0
  cout << "d_nodes->begin() = " << d_nodes->begin() << " d_nodes->end() = " 
       << d_nodes->end() << endl;
#endif

  //  Iterator iii(d_cells);

#if 0
  cout << "Iterator output . . . " << endl;
  for (Iterator ii(d_cells); !ii.done(); ii++) {
    cout << ii << endl;
  }
#endif
  
}


