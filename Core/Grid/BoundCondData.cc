#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BoundCondBase.h>

using namespace Uintah;
#include <iostream>
#include <utility>
using std::cerr;
using std::endl;

BCData::BCData() 
{
}

BCData::~BCData()
{
}

void 
BCData::setBCValues(int mat_id,BoundCondBase* bc)
{
  if ((int)d_data.size() < mat_id + 1)
    d_data.resize(mat_id +1);

  cerr << "Setting bc " << bc->getType() << " for material " << mat_id << endl;
  
  d_data[mat_id][bc->getType()]=bc;
}

BoundCondBase*
BCData::getBCValues(int mat_id,const string& type) const
{
  // The default location for BCs defined for all materials is mat_id = 0.
  // Need to first check the actual mat_id specified.  If this is not found,
  // then will check mat_id = 0 case.  If it isn't found, then return 0.
  map<string,BoundCondBase*>::const_iterator it;   
  for (unsigned int i = 0; i < d_data.size(); i++) {
    for (it = d_data[i].begin(); it != d_data[i].end(); it++) {
      cerr << "mat id: " << i << " BC " << it->first << endl;
    }
  }

  if ((int)d_data.size() <= mat_id ) {
    cerr << "Size of d_data is less than mat_id, must be an ALL" << endl;
    map<string,BoundCondBase*>::const_iterator it1;
    it1 = d_data[0].find(type);
    if (it1 == d_data[0].end())
      return 0;
    else
      return it1->second;
    
  } else {
    map<string,BoundCondBase*>::const_iterator iter;   
    iter = d_data[mat_id].find(type);
    if (iter == d_data[mat_id].end()) {
      cerr << "Can't find " << type << " for material " << mat_id << endl;
      iter = d_data[0].find(type);
      if (iter == d_data[0].end()) {
	cerr << "Can't find " << type << " for material " << 0 << endl;
	return 0;
      } else 
	return iter->second;
    } else
      return iter->second;
  }
	
}


