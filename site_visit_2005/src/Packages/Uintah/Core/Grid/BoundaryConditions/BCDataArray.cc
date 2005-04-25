#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/DifferenceBCData.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/SideBCData.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <algorithm>
#include <set>
#include <vector>
#include <functional>
#include <sgi_stl_warnings_on.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

// export SCI_DEBUG="BCDA_DBG:+"
static DebugStream BCDA_dbg("BCDA_DBG",false);

BCDataArray::BCDataArray() 
{
}

BCDataArray::~BCDataArray()
{
  bcDataArrayType::const_iterator mat_id_itr;
  for (mat_id_itr=d_BCDataArray.begin(); mat_id_itr != d_BCDataArray.end();
       ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    vector<BCGeomBase*>& vec = d_BCDataArray[mat_id];
    vector<BCGeomBase*>::const_iterator bcd_itr;
    for (bcd_itr = vec.begin(); bcd_itr != vec.end(); ++bcd_itr) {
      delete *bcd_itr;
    }
    vec.clear();
  }
  d_BCDataArray.clear();
  
}

BCDataArray::BCDataArray(const BCDataArray& mybc)
{
  bcDataArrayType::const_iterator mat_id_itr;
  for (mat_id_itr = mybc.d_BCDataArray.begin(); 
       mat_id_itr != mybc.d_BCDataArray.end(); ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    const vector<BCGeomBase*>& mybc_vec = mat_id_itr->second;
    vector<BCGeomBase*>& d_BCDataArray_vec =  d_BCDataArray[mat_id];
    vector<BCGeomBase*>::const_iterator vec_itr;
    for (vec_itr = mybc_vec.begin(); vec_itr != mybc_vec.end(); ++vec_itr) {
      d_BCDataArray_vec.push_back((*vec_itr)->clone());
    }
  }
}

BCDataArray& BCDataArray::operator=(const BCDataArray& rhs)
{
  if (this == &rhs) 
    return *this;

  // Delete the lhs
  bcDataArrayType::const_iterator mat_id_itr;
  for (mat_id_itr=d_BCDataArray.begin(); mat_id_itr != d_BCDataArray.end();
       ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    vector<BCGeomBase*>& vec = d_BCDataArray[mat_id];
    vector<BCGeomBase*>::const_iterator bcd_itr;
    for (bcd_itr = vec.begin(); bcd_itr != vec.end(); ++bcd_itr)
	delete *bcd_itr;
    vec.clear();
  }
  d_BCDataArray.clear();
  // Copy the rhs to the lhs
  for (mat_id_itr = rhs.d_BCDataArray.begin(); 
       mat_id_itr != rhs.d_BCDataArray.end(); ++mat_id_itr) {
    int mat_id = mat_id_itr->first;
    vector<BCGeomBase*>& d_BCDataArray_vec = d_BCDataArray[mat_id];
    const vector<BCGeomBase*>& rhs_vec = mat_id_itr->second;
    vector<BCGeomBase*>::const_iterator vec_itr;
    for (vec_itr = rhs_vec.begin(); vec_itr != rhs_vec.end(); ++vec_itr) 
      d_BCDataArray_vec.push_back((*vec_itr)->clone());
  }
  return *this;
}

BCDataArray* BCDataArray::clone()
{
  return scinew BCDataArray(*this);

}

void BCDataArray::determineIteratorLimits(Patch::FaceType face,
					  const Patch* patch)
{
  IntVector lpts,hpts;
  patch->getFaceCells(face,-1,lpts,hpts);
  vector<Point> test_pts;

  for (CellIterator candidatePoints(lpts,hpts); !candidatePoints.done();
       candidatePoints++) {
    IntVector nodes[8];
    patch->findNodesFromCell(*candidatePoints,nodes);
    Point pts[8];
    Vector p;
    for (int i = 0; i < 8; i++)
      pts[i] = patch->getLevel()->getNodePosition(nodes[i]);
    if (face == Patch::xminus)
      p = (pts[0].asVector()+pts[1].asVector()+pts[2].asVector()
	   +pts[3].asVector())/4.;
    if (face == Patch::xplus)
      p = (pts[4].asVector()+pts[5].asVector()+pts[6].asVector()
	   +pts[7].asVector())/4.;
    if (face == Patch::yminus)
      p = (pts[0].asVector()+pts[1].asVector()+pts[4].asVector()
	   +pts[5].asVector())/4.;
    if (face == Patch::yplus)
      p = (pts[2].asVector()+pts[3].asVector()+pts[6].asVector()
	   +pts[7].asVector())/4.;
    if (face == Patch::zminus)
      p = (pts[0].asVector()+pts[2].asVector()+pts[4].asVector()
	   +pts[6].asVector())/4.;
    if (face == Patch::zplus)
      p = (pts[1].asVector()+pts[3].asVector()+pts[5].asVector()
	   +pts[7].asVector())/4.;

    test_pts.push_back(Point(p.x(),p.y(),p.z()));
  }
  
  BCDataArray::bcDataArrayType::iterator mat_id_itr;
  for (mat_id_itr = d_BCDataArray.begin();
       mat_id_itr != d_BCDataArray.end(); ++mat_id_itr) {
    vector<BCGeomBase*>& bc_objects = mat_id_itr->second;
    for (vector<BCGeomBase*>::iterator obj = bc_objects.begin();
	 obj != bc_objects.end(); ++obj) {
      (*obj)->determineIteratorLimits(face,patch,test_pts);
#if 0
      (*obj)->printLimits();
#endif
    }
  }

}

void BCDataArray::addBCData(int mat_id,BCGeomBase* bc)
{
  vector<BCGeomBase*>& d_BCDataArray_vec = d_BCDataArray[mat_id];
  d_BCDataArray_vec.push_back(bc);
}

void BCDataArray::combineBCGeometryTypes(int mat_id)
{
  vector<BCGeomBase*>& d_BCDataArray_vec = d_BCDataArray[mat_id];
  
  vector<BCGeomBase*> new_bcdata_array;
  // Look to see if there are duplicate SideBCData types, if so, then
  // combine them into one (i.e. copy the BCData from the duplicate into
  // the one that will actually be stored).

  if (count_if(d_BCDataArray_vec.begin(),d_BCDataArray_vec.end(),
	       cmp_type<SideBCData>()) > 1) {
    cout << "Have duplicates Before . . ." << endl;
    for (vector<BCGeomBase*>::const_iterator v_itr = d_BCDataArray_vec.begin();
	 v_itr != d_BCDataArray_vec.end(); ++v_itr) {
      (*v_itr)->print();
    }
  }

  if (count_if(d_BCDataArray_vec.begin(),d_BCDataArray_vec.end(),
	       cmp_type<SideBCData>()) > 1) {
    
    SideBCData* side_bc = scinew SideBCData();
    for (vector<BCGeomBase*>::const_iterator itr = d_BCDataArray_vec.begin();
	 itr != d_BCDataArray_vec.end(); ++ itr) {
      if (typeid(*(*itr)) == typeid(SideBCData)) {
	cout << "Found SideBCData" << endl;
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

    cout << "Have duplicates After . . ." << endl;
    for (vector<BCGeomBase*>::const_iterator v_itr = new_bcdata_array.begin();
	 v_itr != new_bcdata_array.end(); ++v_itr) {
      (*v_itr)->print();
    }
    for_each(d_BCDataArray_vec.begin(),d_BCDataArray_vec.end(),
	     delete_object<BCGeomBase>());
    d_BCDataArray_vec.clear();
    d_BCDataArray_vec = new_bcdata_array;
  }
  
}

const BoundCondBase* 
BCDataArray::getBoundCondData(int mat_id, string type, int i) const
{
  BCData new_bc,new_bc_all;
  // Need to check two scenarios -- the given mat_id and the all mat_id (-1)
  // Check the given mat_id
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  
  if (itr != d_BCDataArray.end()) {
    itr->second[i]->getBCData(new_bc);
    bool found_it = new_bc.find(type);
    if (found_it == true)
      return new_bc.getBCValues(type);
  }
  // Check the mat_id = "all" case
  itr = d_BCDataArray.find(-1);
  if (itr  != d_BCDataArray.end()) {
    if (i < (int)itr->second.size()) {
      itr->second[i]->getBCData(new_bc_all);
      bool found_it = new_bc_all.find(type);
      if (found_it == true)
	return new_bc_all.getBCValues(type);
      else
	return 0;
    }
  }
  return 0;
}


void BCDataArray::setBoundaryIterator(int mat_id,vector<IntVector>& b,int i)
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    itr->second[i]->setBoundaryIterator(b);
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[i]->setBoundaryIterator(b);
  }
}

void BCDataArray::setNBoundaryIterator(int mat_id,vector<IntVector>& b,int i)
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    itr->second[i]->setNBoundaryIterator(b);
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[i]->setNBoundaryIterator(b);
  }
}

void BCDataArray::setSFCXIterator(int mat_id,vector<IntVector>& i,int ii)
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    itr->second[ii]->setSFCXIterator(i);
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->setSFCXIterator(i);
  }
}

void BCDataArray::setSFCYIterator(int mat_id,vector<IntVector>& i,int ii)
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    itr->second[ii]->setSFCYIterator(i);
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->setSFCYIterator(i);
  }
}

void BCDataArray::setSFCZIterator(int mat_id,vector<IntVector>& i,int ii)
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    itr->second[ii]->setSFCZIterator(i);
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->setSFCZIterator(i);
  }
}

void BCDataArray::getBoundaryIterator(int mat_id,vector<IntVector>& b,
				      int i) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[i]->getBoundaryIterator(b);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[i]->getBoundaryIterator(b);
  }
}

void BCDataArray::getNBoundaryIterator(int mat_id,vector<IntVector>& b,
				       int i) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[i]->getNBoundaryIterator(b);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[i]->getNBoundaryIterator(b);
  }
}

void BCDataArray::getSFCXIterator(int mat_id,vector<IntVector>& i,int ii) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[ii]->getSFCXIterator(i);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->getSFCXIterator(i);
  }
}

void BCDataArray::getSFCYIterator(int mat_id,vector<IntVector>& i,int ii) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[ii]->getSFCYIterator(i);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->getSFCYIterator(i);
  }
}

void BCDataArray::getSFCZIterator(int mat_id,vector<IntVector>& i,int ii) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[ii]->getSFCZIterator(i);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->getSFCZIterator(i);
  }
}

int BCDataArray::getNumberChildren(int mat_id) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    return itr->second.size();
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      return itr->second.size();
  }
  return 0;
}

BCGeomBase* BCDataArray::getChild(int mat_id,int i) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    return itr->second[i];
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      return itr->second[i];
  }
  return 0;
}

void BCDataArray::print()
{
  bcDataArrayType::const_iterator bcda_itr;
  for (bcda_itr = d_BCDataArray.begin(); bcda_itr != d_BCDataArray.end(); 
       bcda_itr++) {
    BCDA_dbg << endl << "mat_id = " << bcda_itr->first << endl;
    for (vector<BCGeomBase*>::const_iterator i = bcda_itr->second.begin();
	 i != bcda_itr->second.end(); ++i) {
      BCDA_dbg << "BCGeometry Type = " << typeid(*(*i)).name() <<  " "
	   << *i << endl;
      (*i)->print();
    }
  }
	
  
}
