/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/BoundaryConditions/UnstructuredSideBCData.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace Uintah;

UnstructuredSideBCData::UnstructuredSideBCData() 
{
  d_cells=UnstructuredGridIterator(IntVector(0,0,0),IntVector(0,0,0));
  d_nodes=UnstructuredGridIterator(IntVector(0,0,0),IntVector(0,0,0));
  d_bcname = "NotSet"; 
}


UnstructuredSideBCData::~UnstructuredSideBCData()
{
}

bool UnstructuredSideBCData::operator==(const UnstructuredBCGeomBase& rhs) const
{
  const UnstructuredSideBCData* p_rhs = 
    dynamic_cast<const UnstructuredSideBCData*>(&rhs);

  if (p_rhs == nullptr)
    return false;
  else
    return true;
}

UnstructuredSideBCData* UnstructuredSideBCData::clone()
{
  return scinew UnstructuredSideBCData(*this);

}
void UnstructuredSideBCData::addBCData(BCData& bc)
{
  d_bc = bc;
}

void UnstructuredSideBCData::addBC(BoundCondBase* bc)
{
  d_bc.setBCValues(bc);
}

void UnstructuredSideBCData::sudoAddBC(BoundCondBase* bc)
{
  d_bc.setBCValues(bc);
}


void UnstructuredSideBCData::getBCData(BCData& bc) const
{
  bc = d_bc;
}

bool UnstructuredSideBCData::inside(const Point &p) const 
{
  return true;
}

void UnstructuredSideBCData::print()
{
  BC_dbg << "Geometry type = " << typeid(this).name() << std::endl;
  d_bc.print();
}


void UnstructuredSideBCData::determineIteratorLimits(UnstructuredPatch::FaceType face, 
                                         const UnstructuredPatch* patch, 
                                         std::vector<Point>& test_pts)
{
#if 0
  std::cout << "UnstructuredSideBC determineIteratorLimits() " << patch->getFaceName(face)<<  std::endl;
#endif


  IntVector l,h;
  patch->getFaceCells(face,0,l,h);
  d_cells = UnstructuredGridIterator(l,h);

#if 0
  std::cout << "d_cells->begin() = " << d_cells->begin() << " d_cells->end() = " 
       << d_cells->end() << std::endl;
#endif


  IntVector ln,hn;
  patch->getFaceNodes(face,0,ln,hn);
  d_nodes = UnstructuredGridIterator(ln,hn);


#if 0
  std::cout << "d_nodes->begin() = " << d_nodes->begin() << " d_nodes->end() = " 
       << d_nodes->end() << std::endl;
#endif

  //  Iterator iii(d_cells);

#if 0
  std::cout << "Iterator output . . . " << std::endl;
  for (Iterator ii(d_cells); !ii.done(); ii++) {
    std::cout << ii << std::endl;
  }
#endif
  
}


