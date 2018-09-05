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

#include <Core/Grid/BoundaryConditions/UnstructuredUnionBCData.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Grid/Variables/UnstructuredUnionIterator.h>
#include <Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <algorithm>

using namespace Uintah;

UnstructuredUnionBCData::UnstructuredUnionBCData() : UnstructuredBCGeomBase()
{
}

UnstructuredUnionBCData::~UnstructuredUnionBCData()
{
  for (std::vector<UnstructuredBCGeomBase*>::const_iterator bc = child.begin();
       bc != child.end(); ++bc)
    delete (*bc);
  
  child.clear();
}


UnstructuredUnionBCData::UnstructuredUnionBCData(const UnstructuredUnionBCData& mybc): UnstructuredBCGeomBase(mybc)
{
  std::vector<UnstructuredBCGeomBase*>::const_iterator itr;
  for (itr=mybc.child.begin(); itr != mybc.child.end(); ++itr)
    child.push_back((*itr)->clone());
}

UnstructuredUnionBCData& UnstructuredUnionBCData::operator=(const UnstructuredUnionBCData& rhs)
{
  UnstructuredBCGeomBase::operator=(rhs);

  if (this == &rhs)
    return *this;

  // Delete the lhs
  std::vector<UnstructuredBCGeomBase*>::const_iterator itr;
  for(itr=child.begin(); itr != child.end();++itr)
    delete *itr;

  child.clear();
  
  // copy the rhs to the lhs
  for (itr=rhs.child.begin(); itr != rhs.child.end();++itr)
    child.push_back((*itr)->clone());
  
  return *this;
}


bool UnstructuredUnionBCData::operator==(const UnstructuredBCGeomBase& rhs) const
{
  const UnstructuredUnionBCData* p_rhs = 
    dynamic_cast<const UnstructuredUnionBCData*>(&rhs);

  if (p_rhs == nullptr)
    return false;
  else {
    if (this->child.size() != p_rhs->child.size())
      return false;

    return equal(this->child.begin(),this->child.end(),p_rhs->child.begin());
  }
}

UnstructuredUnionBCData* UnstructuredUnionBCData::clone()
{
  return scinew UnstructuredUnionBCData(*this);
}

void UnstructuredUnionBCData::addBCData(BCData& bc)
{
}

void UnstructuredUnionBCData::addBC(BoundCondBase* bc)
{
}

void UnstructuredUnionBCData::sudoAddBC(BoundCondBase* bc)
{
  for (unsigned int i=0 ; i < child.size(); i++)
    child[i]->sudoAddBC(bc);  // or add to zero element only?
}

void UnstructuredUnionBCData::addBCData(UnstructuredBCGeomBase* bc)
{
  child.push_back(bc);
}

void UnstructuredUnionBCData::getBCData(BCData& bc) const
{
  child[0]->getBCData(bc);
}

bool UnstructuredUnionBCData::inside(const Point &p) const 
{
  for (std::vector<UnstructuredBCGeomBase*>::const_iterator i = child.begin(); i != child.end();
       ++i){
    if ((*i)->inside(p))
      return true;
  }
  return false;
}

void UnstructuredUnionBCData::print()
{
  BC_dbg << "Geometry type = " << typeid(this).name() << std::endl;
  for (std::vector<UnstructuredBCGeomBase*>::const_iterator i = child.begin(); i != child.end();
       ++i)
    (*i)->print();

}


void UnstructuredUnionBCData::determineIteratorLimits(UnstructuredPatch::FaceType face, 
                                          const UnstructuredPatch* patch, 
                                          std::vector<Point>& test_pts)
{
#if 0
  cout << "UnstructuredUnionBC determineIteratorLimits()" << endl;
#endif

  for (std::vector<UnstructuredBCGeomBase*>::const_iterator bc = child.begin();
       bc != child.end(); ++bc) {
    (*bc)->determineIteratorLimits(face,patch,test_pts);
  }
  
  UnstructuredUnionIterator cells,nodes;

  for (std::vector<UnstructuredBCGeomBase*>::const_iterator bc = child.begin();
       bc != child.end(); ++bc) {
    Iterator cell_itr,node_itr;
    (*bc)->getCellFaceIterator(cell_itr);
    (*bc)->getNodeFaceIterator(node_itr);
    Iterator base_ci(cells),base_ni(nodes);
    cells = UnstructuredUnionIterator(base_ci,cell_itr);
    nodes = UnstructuredUnionIterator(base_ni,node_itr);
  }

  d_cells = UnstructuredUnionIterator(cells);   
  d_nodes = UnstructuredUnionIterator(nodes); 


#if 0
  IntVector l,h;
  patch->getFaceCells(face,0,l,h);

  std::vector<IntVector> b,nb;
  std::vector<Point>::iterator pts;
  pts = test_pts.begin();
  for (CellIterator bound(l,h); !bound.done(); bound++,pts++) 
    if (inside(*pts))
      b.push_back(*bound);

  setBoundaryIterator(b);
#if 0
  cout << "Size of boundary = " << boundary.size() << endl;
#endif
  // Need to determine the boundary iterators for each separate bc.
  for (std::vector<UnstructuredBCGeomBase*>::const_iterator bc = child.begin();  
       bc != child.end(); ++bc) {
    pts = test_pts.begin();
    std::vector<IntVector> boundary_itr;
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

