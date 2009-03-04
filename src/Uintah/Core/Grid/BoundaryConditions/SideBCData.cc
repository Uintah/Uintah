/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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
  d_cells = GridIterator(l,h);

#if 0
  cout << "d_cells->begin() = " << d_cells->begin() << " d_cells->end() = " 
       << d_cells->end() << endl;
#endif


  IntVector ln,hn;
  patch->getFaceNodes(face,0,ln,hn);
  d_nodes = GridIterator(ln,hn);


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


