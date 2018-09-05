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

#include <Core/Grid/BoundaryConditions/UnstructuredBCDataArray.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Grid/BoundaryConditions/UnstructuredDifferenceBCData.h>
#include <Core/Grid/BoundaryConditions/UnstructuredSideBCData.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/UnstructuredLevel.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <algorithm>
#include <set>
#include <vector>
#include <functional>

using namespace Uintah;
using namespace std;

namespace {
  DebugStream BCDA_dbg("BCDA_DBG", "Grid_BoundaryConditions", "", false);
}

//------------------------------------------------------------------------------------------------

UnstructuredBCDataArray::UnstructuredBCDataArray() 
{
}

//------------------------------------------------------------------------------------------------

UnstructuredBCDataArray::~UnstructuredBCDataArray()
{
 // cout << "Calling UnstructuredBCDataArray destructor" << endl;
  bcDataArrayType::const_iterator mat_id_itr;
  for (mat_id_itr=d_UnstructuredBCDataArray.begin(); mat_id_itr != d_UnstructuredBCDataArray.end();
       ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    vector<UnstructuredBCGeomBase*>& vec = d_UnstructuredBCDataArray[mat_id];
    vector<UnstructuredBCGeomBase*>::const_iterator bcd_itr;
    for (bcd_itr = vec.begin(); bcd_itr != vec.end(); ++bcd_itr) {
      delete *bcd_itr;
    }
    vec.clear();
  }
  d_UnstructuredBCDataArray.clear();
  
}

//------------------------------------------------------------------------------------------------

UnstructuredBCDataArray::UnstructuredBCDataArray(const UnstructuredBCDataArray& mybc)
{
  bcDataArrayType::const_iterator mat_id_itr;
  for (mat_id_itr = mybc.d_UnstructuredBCDataArray.begin(); 
       mat_id_itr != mybc.d_UnstructuredBCDataArray.end(); ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    const vector<UnstructuredBCGeomBase*>& mybc_vec = mat_id_itr->second;
    vector<UnstructuredBCGeomBase*>& d_UnstructuredBCDataArray_vec =  d_UnstructuredBCDataArray[mat_id];
    vector<UnstructuredBCGeomBase*>::const_iterator vec_itr;
    for (vec_itr = mybc_vec.begin(); vec_itr != mybc_vec.end(); ++vec_itr) {
      d_UnstructuredBCDataArray_vec.push_back((*vec_itr)->clone());
    }
  }
}

//------------------------------------------------------------------------------------------------

UnstructuredBCDataArray& UnstructuredBCDataArray::operator=(const UnstructuredBCDataArray& rhs)
{
  if (this == &rhs) 
    return *this;

  // Delete the lhs
  bcDataArrayType::const_iterator mat_id_itr;
  for (mat_id_itr=d_UnstructuredBCDataArray.begin(); mat_id_itr != d_UnstructuredBCDataArray.end();
       ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    vector<UnstructuredBCGeomBase*>& vec = d_UnstructuredBCDataArray[mat_id];
    vector<UnstructuredBCGeomBase*>::const_iterator bcd_itr;
    for (bcd_itr = vec.begin(); bcd_itr != vec.end(); ++bcd_itr)
        delete *bcd_itr;
    vec.clear();
  }
  d_UnstructuredBCDataArray.clear();
  // Copy the rhs to the lhs
  for (mat_id_itr = rhs.d_UnstructuredBCDataArray.begin(); 
       mat_id_itr != rhs.d_UnstructuredBCDataArray.end(); ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    vector<UnstructuredBCGeomBase*>& d_UnstructuredBCDataArray_vec = d_UnstructuredBCDataArray[mat_id];
    const vector<UnstructuredBCGeomBase*>& rhs_vec = mat_id_itr->second;
    vector<UnstructuredBCGeomBase*>::const_iterator vec_itr;
    for (vec_itr = rhs_vec.begin(); vec_itr != rhs_vec.end(); ++vec_itr) 
      d_UnstructuredBCDataArray_vec.push_back((*vec_itr)->clone());
  }
  return *this;
}

UnstructuredBCDataArray* UnstructuredBCDataArray::clone()
{
  return scinew UnstructuredBCDataArray(*this);
}

//------------------------------------------------------------------------------------------------

void UnstructuredBCDataArray::determineIteratorLimits(UnstructuredPatch::FaceType face,
                                          const UnstructuredPatch* patch)
{
  IntVector lpts,hpts;
  patch->getFaceCells(face,-1,lpts,hpts);
  std::vector<Point> test_pts;
  
  for (CellIterator candidatePoints(lpts,hpts); !candidatePoints.done();
       candidatePoints++) {
    IntVector nodes[8];
    patch->findNodesFromCell(*candidatePoints,nodes);
    Point pts[8];
    Vector p( 0.0, 0.0, 0.0 );
    
    for (int i = 0; i < 8; i++)
      pts[i] = patch->getLevel()->getNodePosition(nodes[i]);
    
    if (face == UnstructuredPatch::xminus)
      p = (pts[0].toVector()+pts[1].toVector()+pts[2].toVector()
           +pts[3].toVector())/4.;
    if (face == UnstructuredPatch::xplus)
      p = (pts[4].toVector()+pts[5].toVector()+pts[6].toVector()
           +pts[7].toVector())/4.;
    if (face == UnstructuredPatch::yminus)
      p = (pts[0].toVector()+pts[1].toVector()+pts[4].toVector()
           +pts[5].toVector())/4.;
    if (face == UnstructuredPatch::yplus)
      p = (pts[2].toVector()+pts[3].toVector()+pts[6].toVector()
           +pts[7].toVector())/4.;
    if (face == UnstructuredPatch::zminus)
      p = (pts[0].toVector()+pts[2].toVector()+pts[4].toVector()
           +pts[6].toVector())/4.;
    if (face == UnstructuredPatch::zplus)
      p = (pts[1].toVector()+pts[3].toVector()+pts[5].toVector()
           +pts[7].toVector())/4.;
    
    test_pts.push_back(Point(p.x(),p.y(),p.z()));
  }
  
  UnstructuredBCDataArray::bcDataArrayType::iterator mat_id_itr;
  for (mat_id_itr = d_UnstructuredBCDataArray.begin();
       mat_id_itr != d_UnstructuredBCDataArray.end(); ++mat_id_itr) {
    vector<UnstructuredBCGeomBase*>& bc_objects = mat_id_itr->second;
    for (vector<UnstructuredBCGeomBase*>::iterator obj = bc_objects.begin();
         obj != bc_objects.end(); ++obj) {
      (*obj)->determineIteratorLimits(face,patch,test_pts);
#if 0
      std::cout << "printing domain bc on face: " << face << std::endl;
      (*obj)->printLimits();
#endif
    }
  }
  
  // A BCDataArry contains a bunch of geometry objects. Here, we remove objects with empty iterators.
  // The reason that we get geometry objects with zero iterators is that, for a given boundary face
  // that is shared across several patches, geometry objects are created on ALL patches. Later,
  // a geometry object is assigned an iterator depending on whether it lives on that patch or not.
  for (mat_id_itr = d_UnstructuredBCDataArray.begin();
       mat_id_itr != d_UnstructuredBCDataArray.end(); ++mat_id_itr) {
    vector<UnstructuredBCGeomBase*>& bc_objects = mat_id_itr->second;
    for (vector<UnstructuredBCGeomBase*>::iterator obj = bc_objects.begin();
         obj < bc_objects.end();) {
      if ( !( (*obj)->hasIterator()) ) {
        delete *obj;
        obj = bc_objects.erase(obj); // point the iterator to the next element that was after the one we deleted
      } else {
        ++obj;
      }
    }
  }
}

//------------------------------------------------------------------------------------------------

void UnstructuredBCDataArray::determineInteriorBndIteratorLimits(UnstructuredPatch::FaceType face,
                                          const UnstructuredPatch* patch)
{
  UnstructuredBCDataArray::bcDataArrayType::iterator mat_id_itr;
  for (mat_id_itr = d_UnstructuredBCDataArray.begin();
       mat_id_itr != d_UnstructuredBCDataArray.end(); ++mat_id_itr) {
    vector<UnstructuredBCGeomBase*>& bc_objects = mat_id_itr->second;
    for (vector<UnstructuredBCGeomBase*>::iterator obj = bc_objects.begin();
         obj != bc_objects.end(); ++obj) {
      (*obj)->determineInteriorBndIteratorLimits(face,patch);
#if 0
      (*obj)->printLimits();
#endif
    }
  }

  // A BCDataArry contains a bunch of geometry objects. Here, we remove objects with empty iterators.
  // The reason that we get geometry objects with zero iterators is that, for a given boundary face
  // that is shared across several patches, geometry objects are created on ALL patches. Later,
  // a geometry object is assigned an iterator depending on whether it lives on that patch or not.
  for (mat_id_itr = d_UnstructuredBCDataArray.begin();
       mat_id_itr != d_UnstructuredBCDataArray.end(); ++mat_id_itr) {
    vector<UnstructuredBCGeomBase*>& bc_objects = mat_id_itr->second;
    for (vector<UnstructuredBCGeomBase*>::iterator obj = bc_objects.begin();
         obj < bc_objects.end();) {
      if ( !( (*obj)->hasIterator()) ) {
        delete *obj;
        obj = bc_objects.erase(obj); // point the iterator to the next element that was after the one we deleted
      } else {
        ++obj;
      }
    }
  }
}

//------------------------------------------------------------------------------------------------

void UnstructuredBCDataArray::addBCData(int mat_id,UnstructuredBCGeomBase* bc)
{
  vector<UnstructuredBCGeomBase*>& d_UnstructuredBCDataArray_vec = d_UnstructuredBCDataArray[mat_id];
  d_UnstructuredBCDataArray_vec.push_back(bc);
}

//------------------------------------------------------------------------------------------------

void UnstructuredBCDataArray::combineBCGeometryTypes(int mat_id)
{

  vector<UnstructuredBCGeomBase*>& d_UnstructuredBCDataArray_vec = d_UnstructuredBCDataArray[mat_id];
  
  vector<UnstructuredBCGeomBase*> new_bcdata_array;
  // Look to see if there are duplicate SideBCData types, if so, then
  // combine them into one (i.e. copy the BCData from the duplicate into
  // the one that will actually be stored).

  if (count_if(d_UnstructuredBCDataArray_vec.begin(),d_UnstructuredBCDataArray_vec.end(),
               u_cmp_type<UnstructuredSideBCData>()) > 1) {
    BCDA_dbg<< "Have duplicates Before . . ." << endl;
    for (vector<UnstructuredBCGeomBase*>::const_iterator v_itr = d_UnstructuredBCDataArray_vec.begin();
         v_itr != d_UnstructuredBCDataArray_vec.end(); ++v_itr) {
      (*v_itr)->print();
    }
    BCDA_dbg<< endl << endl;
  }

  if (count_if(d_UnstructuredBCDataArray_vec.begin(),d_UnstructuredBCDataArray_vec.end(),
               u_cmp_type<UnstructuredSideBCData>()) > 1) {
    
    UnstructuredSideBCData* side_bc = scinew UnstructuredSideBCData();
    for (vector<UnstructuredBCGeomBase*>::const_iterator itr = d_UnstructuredBCDataArray_vec.begin();
         itr != d_UnstructuredBCDataArray_vec.end(); ++ itr) {
      if (typeid(*(*itr)) == typeid(UnstructuredSideBCData)) {
        BCDA_dbg<< "Found SideBCData" << endl;
        BCData bcd,s_bcd;
        (*itr)->getBCData(bcd);
        side_bc->getBCData(s_bcd);
        s_bcd.combine(bcd);
        side_bc->addBCData(s_bcd);
        side_bc->print();
      } else {
        new_bcdata_array.push_back((*itr)->clone());
      }
      
    }
    side_bc->print();
    new_bcdata_array.push_back(side_bc->clone());
    delete side_bc;
    new_bcdata_array.back()->print();

    BCDA_dbg<< "Have duplicates After . . ." << endl;
    for (vector<UnstructuredBCGeomBase*>::const_iterator v_itr = new_bcdata_array.begin();
         v_itr != new_bcdata_array.end(); ++v_itr) {
      (*v_itr)->print();
    }
    for_each(d_UnstructuredBCDataArray_vec.begin(),d_UnstructuredBCDataArray_vec.end(),
             u_delete_object<UnstructuredBCGeomBase>());
    d_UnstructuredBCDataArray_vec.clear();
    d_UnstructuredBCDataArray_vec = new_bcdata_array;
  }
  
}

//------------------------------------------------------------------------------------------------

void UnstructuredBCDataArray::combineBCGeometryTypes_NEW(int mat_id)
{
  if (d_UnstructuredBCDataArray[mat_id].size() <= 1) {
    BCDA_dbg<< "One or fewer elements in UnstructuredBCDataArray" << endl << endl;
    return;
  }

  vector<UnstructuredBCGeomBase*>& d_UnstructuredBCDataArray_vec = d_UnstructuredBCDataArray[mat_id];
  
  vector<UnstructuredBCGeomBase*> new_bcdata_array;
  // Look to see if there are duplicate SideBCData types, if so, then
  // combine them into one (i.e. copy the BCData from the duplicate into
  // the one that will actually be stored).

  //  count the number of unique geometry types

  vector<UnstructuredBCGeomBase*>::iterator v_itr,nv_itr;

  for (v_itr = d_UnstructuredBCDataArray_vec.begin(); v_itr != d_UnstructuredBCDataArray_vec.end(); 
       ++v_itr) {
    BCDA_dbg<< "number of SideBCData = " << 
      count_if(d_UnstructuredBCDataArray_vec.begin(),d_UnstructuredBCDataArray_vec.end(),
               u_cmp_type<UnstructuredSideBCData>()) << endl;
  }


  if (count_if(d_UnstructuredBCDataArray_vec.begin(),d_UnstructuredBCDataArray_vec.end(),
               u_cmp_type<UnstructuredSideBCData>()) > 1) {
    BCDA_dbg<< "Have duplicates Before . . ." << endl;
    for (v_itr = d_UnstructuredBCDataArray_vec.begin(); v_itr != d_UnstructuredBCDataArray_vec.end(); 
         ++v_itr) {
      BCDA_dbg<< "type of element = " << typeid(*(*v_itr)).name() << endl;
      (*v_itr)->print();
    }
    BCDA_dbg<< endl << endl;
  }

  // Put the last element in the d_UnstructuredBCDataArray_vec into the new_bcdata_array
  // and delete this element

  UnstructuredBCGeomBase* element = d_UnstructuredBCDataArray_vec.back();
  UnstructuredBCGeomBase* clone_element = element->clone();
  
  new_bcdata_array.push_back(clone_element);
  delete element;
  d_UnstructuredBCDataArray_vec.pop_back();

  while (!d_UnstructuredBCDataArray_vec.empty()){
    element = d_UnstructuredBCDataArray_vec.back();
    bool foundit = false;
    for (nv_itr = new_bcdata_array.begin(); nv_itr != new_bcdata_array.end(); 
         ++nv_itr) {
      if (*(*nv_itr) == *element) {
        foundit = true;
        break;
      }
    }
    if (foundit) {
      d_UnstructuredBCDataArray_vec.pop_back();
      BCData bcd, n_bcd;
      element->getBCData(bcd);
      (*nv_itr)->getBCData(n_bcd);
      n_bcd.combine(bcd);
      (*nv_itr)->addBCData(n_bcd);
      delete element;
    } else {
      new_bcdata_array.push_back(element->clone());
      d_UnstructuredBCDataArray_vec.pop_back();
      delete element;
    }

  }

  BCDA_dbg<< "size of new_bcdata_array = " << new_bcdata_array.size() << endl;
  BCDA_dbg<< "size of d_UnstructuredBCDataArray_vec = " << d_UnstructuredBCDataArray_vec.size() << endl;

  for (nv_itr = new_bcdata_array.begin(); nv_itr != new_bcdata_array.end(); 
       ++nv_itr) {
    (*nv_itr)->print();
    BCDA_dbg<< endl << endl;
  }

  for_each(d_UnstructuredBCDataArray_vec.begin(),d_UnstructuredBCDataArray_vec.end(),
           u_delete_object<UnstructuredBCGeomBase>());

  d_UnstructuredBCDataArray_vec.clear();
#if 1
  d_UnstructuredBCDataArray_vec = new_bcdata_array;
#endif

}

//------------------------------------------------------------------------------------------------

const BoundCondBase* 
UnstructuredBCDataArray::getBoundCondData( int mat_id, const string & type, int ichild ) const
{
  //  cout << "type = " << type << endl;
  BCData new_bc,new_bc_all;
  // Need to check two scenarios -- the given mat_id and the all mat_id (-1)
  // Check the given mat_id
  bcDataArrayType::const_iterator itr = d_UnstructuredBCDataArray.find(mat_id);
  
  if (itr != d_UnstructuredBCDataArray.end()) {
    itr->second[ichild]->getBCData(new_bc);
    bool found_it = new_bc.find(type);
    if (found_it == true)
      return new_bc.cloneBCValues(type);
//    else {
//      found_it = new_bc.find("Auxiliary");
//      if (found_it)
//        return cloneBCValues("Auxiliary");
//    }
  }
  // Check the mat_id = "all" case
  itr = d_UnstructuredBCDataArray.find(-1);
  if (itr  != d_UnstructuredBCDataArray.end()) {
    if (ichild < (int)itr->second.size()) {
      itr->second[ichild]->getBCData(new_bc_all);
      bool found_it = new_bc_all.find(type);
      if (found_it == true)
        return new_bc_all.cloneBCValues(type);
//      else {
//        found_it = new_bc_all.find("Auxiliary");
//        if (found_it)
//          return new_bc_all.cloneBCValues("Auxiliary");
//      }
//      return 0;
    }
  }
  return 0;
}

bool 
UnstructuredBCDataArray::checkForBoundCondData( int & mat_id, const string & type, int ichild ) 
{
  BCData new_bc,new_bc_all;
  // Need to check two scenarios -- the given mat_id and the all mat_id (-1)
  // will update mat_id, to represent the applicable material.
  // Check the given mat_id
  bcDataArrayType::const_iterator itr = d_UnstructuredBCDataArray.find(mat_id);
  if (itr != d_UnstructuredBCDataArray.end()) {
    itr->second[ichild]->getBCData(new_bc);
    bool found_it = new_bc.find(type);
    if (found_it == true){
      return true;
    }
  }
  // Check the mat_id = "all" case
  itr = d_UnstructuredBCDataArray.find(-1);
  if (itr  != d_UnstructuredBCDataArray.end()) {
    if (ichild < (int)itr->second.size()) {
      itr->second[ichild]->getBCData(new_bc_all);
      bool found_it = new_bc_all.find(type);
      if (found_it == true){
        mat_id=-1;
        return true;
      }
    }
  }
  return false;
}

//------------------------------------------------------------------------------------------------

void
UnstructuredBCDataArray::getCellFaceIterator( int mat_id, Iterator & b_ptr, int ichild ) const
{
  bcDataArrayType::const_iterator itr = d_UnstructuredBCDataArray.find(mat_id);
  if (itr != d_UnstructuredBCDataArray.end()) {
    itr->second[ichild]->getCellFaceIterator(b_ptr);
  }
  else {
    itr = d_UnstructuredBCDataArray.find(-1);
    if (itr != d_UnstructuredBCDataArray.end())
      itr->second[ichild]->getCellFaceIterator(b_ptr);
  }

}

//------------------------------------------------------------------------------------------------

void UnstructuredBCDataArray::getNodeFaceIterator(int mat_id, Iterator& b_ptr, int ichild) const
{
  bcDataArrayType::const_iterator itr = d_UnstructuredBCDataArray.find(mat_id);
  if (itr != d_UnstructuredBCDataArray.end()) {
    itr->second[ichild]->getNodeFaceIterator(b_ptr);
  }
  else {
    itr = d_UnstructuredBCDataArray.find(-1);
    if (itr != d_UnstructuredBCDataArray.end())
      itr->second[ichild]->getNodeFaceIterator(b_ptr);
  }
}

//------------------------------------------------------------------------------------------------

int UnstructuredBCDataArray::getNumberChildren(int mat_id) const
{
  bcDataArrayType::const_iterator itr = d_UnstructuredBCDataArray.find(mat_id);
  if (itr != d_UnstructuredBCDataArray.end())
    return itr->second.size();
  else {
    itr = d_UnstructuredBCDataArray.find(-1);
    if (itr != d_UnstructuredBCDataArray.end())
      return itr->second.size();
  }
  return 0;
}

UnstructuredBCGeomBase* UnstructuredBCDataArray::getChild(int mat_id,int i) const
{
  bcDataArrayType::const_iterator itr = d_UnstructuredBCDataArray.find(mat_id);
  if (itr != d_UnstructuredBCDataArray.end())
    return itr->second[i];
  else {
    itr = d_UnstructuredBCDataArray.find(-1);
    if (itr != d_UnstructuredBCDataArray.end())
      return itr->second[i];
  }
  return 0;
}

//------------------------------------------------------------------------------------------------

void UnstructuredBCDataArray::print() const
{
  bcDataArrayType::const_iterator bcda_itr;
  for (bcda_itr = d_UnstructuredBCDataArray.begin(); bcda_itr != d_UnstructuredBCDataArray.end(); 
       bcda_itr++) {
    BCDA_dbg << endl << "mat_id = " << bcda_itr->first << endl;
    BCDA_dbg<< "Size of UnstructuredBCGeomBase vector = " << bcda_itr->second.size() << endl;
    for (vector<UnstructuredBCGeomBase*>::const_iterator i = bcda_itr->second.begin();
         i != bcda_itr->second.end(); ++i) {
      BCDA_dbg << "BCGeometry Type = " << typeid(*(*i)).name() <<  " "
           << *i << endl;
      (*i)->print();
    }
  }
}

//------------------------------------------------------------------------------------------------
