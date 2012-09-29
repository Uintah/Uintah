/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Grid/Variables/ListOfCellsIterator.h>
#include <Core/Grid/Variables/GridIterator.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <iostream>
#include <vector>


using namespace SCIRun;
using namespace Uintah;


BCGeomBase::BCGeomBase() 
{
  d_cells = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
  d_nodes = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
  d_bcname = "NotSet"; 
}


BCGeomBase::BCGeomBase(const BCGeomBase& rhs)
{
  d_cells=rhs.d_cells;
  d_nodes=rhs.d_nodes;
  d_bcname = rhs.d_bcname; 
}


BCGeomBase& BCGeomBase::operator=(const BCGeomBase& rhs)
{
  if (this == &rhs)
    return *this;

  d_cells = rhs.d_cells;
  d_nodes = rhs.d_nodes;
  d_bcname = rhs.d_bcname; 

  return *this;
}


BCGeomBase::~BCGeomBase()
{
}


void BCGeomBase::getCellFaceIterator(Iterator& b_ptr)
{
  b_ptr = d_cells;
}


void BCGeomBase::getNodeFaceIterator(Iterator& b_ptr)
{
  b_ptr = d_nodes;
}


void BCGeomBase::determineIteratorLimits(Patch::FaceType face, 
                                         const Patch* patch, 
                                         vector<Point>& test_pts)
{
#if 0
  cout << "BCGeomBase determineIteratorLimits() " << patch->getFaceName(face)<< endl;
#endif

  IntVector l,h;
  patch->getFaceCells(face,0,l,h);
  GridIterator cells(l,h);


  IntVector ln,hn;
  patch->getFaceNodes(face,0,ln,hn);
  GridIterator nodes(ln,hn);


  Iterator cell_itr(cells), node_itr(nodes);

  vector<Point>::const_iterator pts = test_pts.begin();

  ListOfCellsIterator list_cells;
  vector<IntVector> vec_cells;

  for (cell_itr.reset(); !cell_itr.done();cell_itr++,pts++) {
    if (inside(*pts)) {
      vec_cells.push_back(*cell_itr);
    }
  }
  
  ListOfCellsIterator list_nodes;
  vector<IntVector> vec_nodes;

  for (node_itr.reset(); !node_itr.done();node_itr++) {
    Point p = patch->getLevel()->getNodePosition(*node_itr);
    if (inside(p)) {
      vec_nodes.push_back(*node_itr);
    }
  }

  if (vec_cells.empty()) {
    d_cells = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
  }
  else {
    for (vector<IntVector>::const_iterator i = vec_cells.begin(); 
         i != vec_cells.end(); ++i) {
      list_cells.add(*i);
    }
    d_cells = list_cells;
  }
  if (vec_nodes.empty()) {
    d_nodes = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
  }
  else {
    for (vector<IntVector>::const_iterator i = vec_nodes.begin();
         i != vec_nodes.end(); ++i) {
      list_nodes.add(*i);
    }
    d_nodes = list_nodes;
  }
}



void BCGeomBase::printLimits() const
{
  using namespace std;
  cout << endl;
  cout << "d_cells = " << d_cells.begin() << " " << d_cells.end() << endl;
  cout << "d_nodes = " << d_nodes.begin() << " " << d_nodes.end() << endl;
  cout << endl;

}
