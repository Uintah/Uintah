/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <Core/Grid/BoundaryConditions/UnionBCData.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Grid/Variables/UnionIterator.h>
#include <Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <iostream>
#include <algorithm>

using namespace std;

using namespace SCIRun;
using namespace Uintah;

// export SCI_DEBUG="BC_dbg:+"
static DebugStream BC_dbg("BC_dbg",false);

UnionBCData::UnionBCData() : BCGeomBase()
{
}

UnionBCData::~UnionBCData()
{
  for (vector<BCGeomBase*>::const_iterator bc = child.begin();
       bc != child.end(); ++bc)
    delete (*bc);
  
  child.clear();
}


UnionBCData::UnionBCData(const UnionBCData& mybc): BCGeomBase(mybc)
{

  vector<BCGeomBase*>::const_iterator itr;
  for (itr=mybc.child.begin(); itr != mybc.child.end(); ++itr)
    child.push_back((*itr)->clone());
  
}

UnionBCData& UnionBCData::operator=(const UnionBCData& rhs)
{
  BCGeomBase::operator=(rhs);

  if (this == &rhs)
    return *this;

  // Delete the lhs
  vector<BCGeomBase*>::const_iterator itr;
  for(itr=child.begin(); itr != child.end();++itr)
    delete *itr;

  child.clear();
  
  // copy the rhs to the lhs
  for (itr=rhs.child.begin(); itr != rhs.child.end();++itr)
    child.push_back((*itr)->clone());
  
  return *this;
}


bool UnionBCData::operator==(const BCGeomBase& rhs) const
{
  const UnionBCData* p_rhs = 
    dynamic_cast<const UnionBCData*>(&rhs);

  if (p_rhs == NULL)
    return false;
  else {
    if (this->child.size() != p_rhs->child.size())
      return false;

    return equal(this->child.begin(),this->child.end(),p_rhs->child.begin());
  }

}

UnionBCData* UnionBCData::clone()
{
  return scinew UnionBCData(*this);

}

void UnionBCData::addBCData(BCData& bc)
{
}

void UnionBCData::addBC(BoundCondBase* bc)
{
  
}

void UnionBCData::addBCData(BCGeomBase* bc)
{
  child.push_back(bc);
}

void UnionBCData::getBCData(BCData& bc) const
{
  child[0]->getBCData(bc);
}

bool UnionBCData::inside(const Point &p) const 
{
  for (vector<BCGeomBase*>::const_iterator i = child.begin(); i != child.end();
       ++i){
    if ((*i)->inside(p))
      return true;
  }
  return false;
}

void UnionBCData::print()
{
  BC_dbg << "Geometry type = " << typeid(this).name() << endl;
  for (vector<BCGeomBase*>::const_iterator i = child.begin(); i != child.end();
       ++i)
    (*i)->print();

}


void UnionBCData::determineIteratorLimits(Patch::FaceType face, 
                                          const Patch* patch, 
                                          vector<Point>& test_pts)
{
#if 0
  cout << "UnionBC determineIteratorLimits()" << endl;
#endif

  for (vector<BCGeomBase*>::const_iterator bc = child.begin();
       bc != child.end(); ++bc) {
    (*bc)->determineIteratorLimits(face,patch,test_pts);
  }
  
  UnionIterator cells,nodes;

  for (vector<BCGeomBase*>::const_iterator bc = child.begin();
       bc != child.end(); ++bc) {
    Iterator cell_itr,node_itr;
    (*bc)->getCellFaceIterator(cell_itr);
    (*bc)->getNodeFaceIterator(node_itr);
    Iterator base_ci(cells),base_ni(nodes);
    cells = UnionIterator(base_ci,cell_itr);
    nodes = UnionIterator(base_ni,node_itr);
  }

  d_cells = UnionIterator(cells);   
  d_nodes = UnionIterator(nodes); 


#if 0
  IntVector l,h;
  patch->getFaceCells(face,0,l,h);

  vector<IntVector> b,nb;
  vector<Point>::iterator pts;
  pts = test_pts.begin();
  for (CellIterator bound(l,h); !bound.done(); bound++,pts++) 
    if (inside(*pts))
      b.push_back(*bound);

  setBoundaryIterator(b);
#if 0
  cout << "Size of boundary = " << boundary.size() << endl;
#endif
  // Need to determine the boundary iterators for each separate bc.
  for (vector<BCGeomBase*>::const_iterator bc = child.begin();  
       bc != child.end(); ++bc) {
    pts = test_pts.begin();
    vector<IntVector> boundary_itr;
    for (CellIterator bound(l,h); !bound.done(); bound++, pts++) 
      if ( (*bc)->inside(*pts))
        boundary_itr.push_back(*bound);
#if 0
    cout << "Size of boundary_itr = " << boundary_itr.size() << endl;
#endif
    (*bc)->setBoundaryIterator(boundary_itr);
  }
    
  IntVector ln,hn;
  patch->getFaceNodes(face,0,ln,hn);
  for (NodeIterator bound(ln,hn);!bound.done();bound++) {
    Point p = patch->getLevel()->getNodePosition(*bound);
    if (inside(p)) 
      nb.push_back(*bound);
  }
  
  setNBoundaryIterator(nb);

#endif

}

