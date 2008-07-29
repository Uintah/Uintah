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
#ifdef OLD
  d_cells = 0;
  d_nodes = 0;
#else
  d_cells = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
  d_nodes = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
#endif
}

#ifdef OLD
BCGeomBase::BCGeomBase(const BCGeomBase& rhs)
{
  if (rhs.d_cells)
    d_cells = rhs.d_cells->clone();
  else
    d_cells = 0;
  if (rhs.d_nodes)
    d_nodes = rhs.d_nodes->clone();
  else
    d_nodes  = 0;

}


BCGeomBase& BCGeomBase::operator=(const BCGeomBase& rhs)
{
  if (this == &rhs)
    return *this;

  // Delete the old values
  delete d_cells;
  delete d_nodes;

  // Copy the rhs to the lhs
  if (rhs.d_cells)
    d_cells = rhs.d_cells->clone();
  else
    d_cells = 0;

  if (rhs.d_nodes)
    d_nodes = rhs.d_nodes->clone();
  else
    d_nodes = 0;


  return *this;
}


BCGeomBase::~BCGeomBase()
{
  if (d_cells)
    delete d_cells;
  
  if (d_nodes)
    delete d_nodes;
}

#else

BCGeomBase::~BCGeomBase()
{
}

#endif

void BCGeomBase::getCellFaceIterator(Iterator& b_ptr)
{
  //Iterator i = Iterator(d_cells);
  //  b_ptr = Iterator(d_cells);
#ifdef OLD
  Iterator itr(*d_cells);
  b_ptr = itr;
#else
  b_ptr = d_cells;
#endif
}


void BCGeomBase::getNodeFaceIterator(Iterator& b_ptr)
{
  //  b_ptr = Iterator(d_nodes);
#ifdef OLD
  Iterator itr(*d_nodes);
  b_ptr = itr;
#else
  b_ptr = d_nodes;
#endif
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
#ifdef OLD
  GridIterator* cells  = scinew GridIterator(l,h);
#else
  GridIterator cells(l,h);
#endif

  IntVector ln,hn;
  patch->getFaceNodes(face,0,ln,hn);
#ifdef OLD
  GridIterator* nodes = scinew GridIterator(ln,hn);
#else
  GridIterator nodes(ln,hn);
#endif

#ifdef OLD
  Iterator cell_itr(*cells), node_itr(*nodes);
#else
  Iterator cell_itr(cells), node_itr(nodes);
#endif
  vector<Point>::const_iterator pts = test_pts.begin();

#ifdef OLD
  ListOfCellsIterator* list_cells = scinew ListOfCellsIterator();
#else
  ListOfCellsIterator list_cells;
#endif
  vector<IntVector> vec_cells;

  for (cell_itr.reset(); !cell_itr.done();cell_itr++,pts++) {
    if (inside(*pts)) {
      //        list_cells->add(*cell_itr);
      vec_cells.push_back(*cell_itr);
    }
  }
  
#ifdef OLD
  ListOfCellsIterator* list_nodes = scinew ListOfCellsIterator();
#else
  ListOfCellsIterator list_nodes;
#endif
  vector<IntVector> vec_nodes;

  for (node_itr.reset(); !node_itr.done();node_itr++) {
    Point p = patch->getLevel()->getNodePosition(*node_itr);
    if (inside(p)) {
      //      list_nodes->add(*node_itr);
      vec_nodes.push_back(*node_itr);
    }
  }

  if (vec_cells.empty()) {
#ifdef OLD
    delete list_cells;
    d_cells = scinew GridIterator(IntVector(0,0,0),IntVector(0,0,0));
#else
    d_cells = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
#endif

  }
  else {
    for (vector<IntVector>::const_iterator i = vec_cells.begin(); 
         i != vec_cells.end(); ++i) {
#ifdef OLD
      list_cells->add(*i);
#else
      list_cells.add(*i);
#endif
    }
    d_cells = list_cells;
  }
  if (vec_nodes.empty()) {
#ifdef OLD
    delete list_nodes;
    d_nodes = scinew GridIterator(IntVector(0,0,0),IntVector(0,0,0));
#else
    d_nodes = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
#endif

  }
  else {
    for (vector<IntVector>::const_iterator i = vec_nodes.begin();
         i != vec_nodes.end(); ++i) {
#ifdef OLD
      list_nodes->add(*i);
#else
      list_nodes.add(*i);
#endif
      
    }
    d_nodes = list_nodes;
  }
#ifdef OLD
  delete cells;
  delete nodes;
#endif

#if 0
  if (d_cells) {
    cout << "d_cells->begin() = " << d_cells->begin() << " d_cells->end() = " 
         << d_cells->end() << endl;
  } else {
    cout << "d_cells is NULL" << endl;
  }

  if (d_nodes) {
    cout << "d_nodes->begin() = " << d_nodes->begin() << " d_nodes->end() = " 
         << d_nodes->end() << endl;
  } else {
    cout << "d_nodes is NULL" << endl;
  }
#endif

#if 0
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
#endif


}



void BCGeomBase::printLimits() const
{
  cout << endl;
#ifdef OLD
  cout << "d_cells = " << d_cells->begin() << " " << d_cells->end() << endl;
  cout << "d_nodes = " << d_nodes->begin() << " " << d_nodes->end() << endl;
#else
  cout << "d_cells = " << d_cells.begin() << " " << d_cells.end() << endl;
  cout << "d_nodes = " << d_nodes.begin() << " " << d_nodes.end() << endl;
#endif
  cout << endl;

}
