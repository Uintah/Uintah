#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BoundCondBase.h>

using namespace Uintah;
#include <iostream>
#include <utility>
using std::cerr;
using std::endl;

BoundCondData::BoundCondData() 
{
}

BoundCondData::~BoundCondData()
{
  boundcondDataType::iterator itr;
  for (itr = d_data.begin(); itr != d_data.end(); itr++) {
    map<string,BoundCondBase*>::const_iterator data_itr;
    for (data_itr=itr->second.begin();data_itr!=itr->second.end();data_itr++){
      delete data_itr->second;
    }
    itr->second.clear();
  }
  d_data.clear();
}

BoundCondData::BoundCondData(const BoundCondData& rhs)
{
  boundcondDataType::const_iterator itr;
  for (itr = rhs.d_data.begin(); itr != rhs.d_data.end(); itr++) {
    map<string, BoundCondBase*>::const_iterator data_itr;
    map<string,BoundCondBase*>& this_data = d_data[itr->first];
    for (data_itr=itr->second.begin(); data_itr!=itr->second.end();data_itr++)
      this_data[data_itr->first] = data_itr->second->clone();
  }
}


BoundCondData& BoundCondData::operator=(const BoundCondData& rhs)
{
  if (this == &rhs)
    return *this;
  
  typedef map<int, map<string,BoundCondBase*> > boundcondDataType;
  boundcondDataType::iterator itr;
  for (itr = d_data.begin(); itr != d_data.end(); itr++) {
    map<string,BoundCondBase*>::const_iterator data_itr;
    for (data_itr=itr->second.begin();data_itr!=itr->second.end(); data_itr++)
      delete data_itr->second;
    itr->second.clear();
  }
  d_data.clear();
 
  boundcondDataType::const_iterator rhs_itr;
  for (rhs_itr = rhs.d_data.begin(); rhs_itr != rhs.d_data.end(); rhs_itr++) {
    map<string,BoundCondBase*>::const_iterator data_itr;
    for (data_itr=rhs_itr->second.begin(); data_itr!=rhs_itr->second.end();
	 data_itr++)
      d_data[rhs_itr->first][data_itr->first] = data_itr->second->clone();
  }
  
  return *this;
}


void 
BoundCondData::setBCValues(int mat_id,BoundCondBase* bc)
{
#if 0
  cerr << "Setting bc " << bc->getType() << " for material " << mat_id << endl;
#endif
  d_data[mat_id][bc->getType()]=bc->clone();
}

#if 0
const BoundCondBase*
BoundCondData::getBCValues(int mat_id,const string& type) const
{
  // The default location for BCs defined for all materials is mat_id = 0.
  // Need to first check the actual mat_id specified.  If this is not found,
  // then will check mat_id = 0 case.  If it isn't found, then return 0.

#if 1
  cerr << "Size of d_data = " << d_data.size() << endl;
  boundcondDataType::const_iterator mat_id_it;   
  map<string,Handle<BoundCondBase> >::const_iterator bc_it;   
  for (mat_id_it = d_data.begin(); mat_id_it != d_data.end(); ++mat_id_it) {
    for (bc_it=mat_id_it->second.begin(); bc_it != mat_id_it->second.end(); 
	 ++bc_it) {
      cerr << "mat id: " << mat_id_it->first << " BC " << bc_it->first << endl;
    }
  }
#endif

  if ((int)d_data.size() == 0)
    return 0;


  boundcondDataType *m = const_cast<boundcondDataType *>(&d_data);
  
  if ((int)d_data.size() <= mat_id ) {
#if 0
    cerr << "Size of d_data is less than mat_id, must be an ALL" << endl;
#endif
    map<string,Handle<BoundCondBase> >::const_iterator it1;
    it1 = (*m)[0].find(type);
    //  it1 = d_data[0].find(type);
    //  if (it1 == d_data[0].end())
    if (it1 == (*m)[0].end())
      return 0;
    else
      return it1->second.get_rep();
    
  } else {
    map<string,Handle<BoundCondBase> >::const_iterator iter;   
    // iter = d_data[mat_id].find(type);
    iter = (*m)[mat_id].find(type);
    //   if (iter == d_data[mat_id].end()) {
    if (iter == (*m)[mat_id].end()) {
#if 0
      cerr << "Can't find " << type << " for material " << mat_id << endl;
#endif
      //    iter = d_data[0].find(type);
      iter = (*m)[0].find(type);
      //      if (iter == d_data[0].end()) {
      if (iter == (*m)[0].end()) {
#if 0
	cerr << "Can't find " << type << " for material " << 0 << endl;
#endif
	return 0;
      } else 
	return iter->second.get_rep();
    } else
      return iter->second.get_rep();
  }
	
}


#else
const BoundCondBase*
BoundCondData::getBCValues(int mat_id,const string& type) const
{
  // The default location for BCs defined for all materials is mat_id = -1.
  // Need to first check the actual mat_id specified.  If this is not found,
  // then will check mat_id = -1 case.  If it isn't found, then return 0.

  boundcondDataType *m = const_cast<boundcondDataType *>(&d_data);

#if 0
  cerr << "Size of d_data = " << d_data.size() << endl;
  boundcondDataType::const_iterator mat_id_it;
  map<string,BoundCondBase* >::const_iterator bc_it;   
  for (mat_id_it = d_data.begin(); mat_id_it != d_data.end(); ++mat_id_it) {
    for (bc_it=mat_id_it->second.begin(); bc_it != mat_id_it->second.end(); 
	 ++bc_it) {
      cerr << "mat id: " << mat_id_it->first << " BC " << bc_it->first << endl;
    }
  }
#endif

  map<string,BoundCondBase* >::const_iterator iter_mat,iter_all;   
  iter_mat = (*m)[mat_id].find(type);

  if (iter_mat == (*m)[mat_id].end()) {
#if 0
    cerr << "Can't find " << type << " for material " << mat_id << endl;
    cerr << "Checking mat_id = all" << endl;
#endif
    // Check the mat_id = -1 (mat all)
    iter_all = (*m)[-1].find(type);
    if (iter_all == (*m)[-1].end()) {
#if 0
      cerr << "Can't find " << type << " for material all" << endl;
#endif
      return 0;
    } else 
      return iter_all->second->clone();
  } else
    return iter_mat->second->clone();
 
}

#endif

int BoundCondData::getMatID() const
{
  boundcondDataType::const_iterator mat_id_it;
  int mat_id = 0;
  for (mat_id_it = d_data.begin(); mat_id_it != d_data.end(); ++mat_id_it) {
    map<string,BoundCondBase* >::const_iterator bc_it;
    for (bc_it=mat_id_it->second.begin(); bc_it != mat_id_it->second.end(); 
	 ++bc_it) { 
      cerr << "mat id: " << mat_id_it->first << " BC " << bc_it->first << endl;
      if (bc_it->first != "Pressure")
	mat_id = mat_id_it->first;
    }
    cerr << "Mat id: " << mat_id  << endl;
  }
  return mat_id;

}
