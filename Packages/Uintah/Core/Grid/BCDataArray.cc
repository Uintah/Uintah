#include <Packages/Uintah/Core/Grid/BCDataArray.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>
#include <iostream>
#include <algorithm>

using namespace SCIRun;
using namespace Uintah;
using std::cout;
using std::endl;
using std::find;

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
    bcDataArrayType* m = const_cast<bcDataArrayType *>(&(mybc.d_BCDataArray));
    vector<BCGeomBase*>& mybc_vec = (*m)[mat_id];
    vector<BCGeomBase*>& d_BCDataArray_vec =  d_BCDataArray[mat_id];
    vector<BCGeomBase*>::iterator vec_itr;
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
    bcDataArrayType* m = const_cast<bcDataArrayType *>(&(rhs.d_BCDataArray));
    vector<BCGeomBase*>& rhs_vec = (*m)[mat_id];
    vector<BCGeomBase*>::iterator vec_itr;
    for (vec_itr = rhs_vec.begin(); vec_itr != rhs_vec.end(); ++vec_itr) 
      d_BCDataArray_vec.push_back((*vec_itr)->clone());
  }
  return *this;
}

BCDataArray* BCDataArray::clone()
{
  return new BCDataArray(*this);

}

void BCDataArray::addBCData(int mat_id,BCGeomBase* bc)
{
  vector<BCGeomBase*>& d_BCDataArray_vec = d_BCDataArray[mat_id];
  d_BCDataArray_vec.push_back(bc);
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


void BCDataArray::setInteriorIterator(int mat_id,vector<IntVector>& i,int ii)
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end())
    itr->second[ii]->setInteriorIterator(i);
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->setInteriorIterator(i);
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

void BCDataArray::getInteriorIterator(int mat_id,vector<IntVector>& i,
				      int ii) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[ii]->getInteriorIterator(i);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ii]->getInteriorIterator(i);
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
}

void BCDataArray::print()
{
  bcDataArrayType::const_iterator bcda_itr;
  for (bcda_itr = d_BCDataArray.begin(); bcda_itr != d_BCDataArray.end(); 
       bcda_itr++) {
    cout << endl << "mat_id = " << bcda_itr->first << endl;
    const vector<BCGeomBase*>& array_vec = bcda_itr->second;
    for (int i = 0; i < (int)array_vec.size(); i++) {
      BCData bcd;
      cout << "BCGeometry Type = " << typeid(*(array_vec[i])).name() <<  " "
	   << array_vec[i] << endl;
      array_vec[i]->getBCData(bcd);
      bcd.print();
    }
  }
	
  
}
