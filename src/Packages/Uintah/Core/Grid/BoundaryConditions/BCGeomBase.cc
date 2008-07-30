#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Packages/Uintah/Core/Grid/Variables/ListOfCellsIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/GridIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <iostream>
#include <vector>


using namespace SCIRun;
using namespace Uintah;


BCGeomBase::BCGeomBase() 
{
  d_cells = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
  d_nodes = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
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
  cout << "BCGeomBase determineIteratorLimits()" << endl;
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
      //        list_cells->add(*cell_itr);
      vec_cells.push_back(*cell_itr);
    }
  }
  
  ListOfCellsIterator list_nodes;
  vector<IntVector> vec_nodes;

  for (node_itr.reset(); !node_itr.done();node_itr++) {
    Point p = patch->getLevel()->getNodePosition(*node_itr);
    if (inside(p)) {
      //      list_nodes->add(*node_itr);
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
  cout << endl;
  cout << "d_cells = " << d_cells.begin() << " " << d_cells.end() << endl;
  cout << "d_nodes = " << d_nodes.begin() << " " << d_nodes.end() << endl;
  cout << endl;

}
