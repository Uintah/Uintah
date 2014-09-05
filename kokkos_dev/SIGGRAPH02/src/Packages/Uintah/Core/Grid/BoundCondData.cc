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
#if 0
  cerr << "Setting bc " << bc->getType() << " for material " << mat_id << endl;
#endif
  d_data[mat_id][bc->getType()]=bc;
}

const BoundCondBase*
BCData::getBCValues(int mat_id,const string& type) const
{
  // The default location for BCs defined for all materials is mat_id = 0.
  // Need to first check the actual mat_id specified.  If this is not found,
  // then will check mat_id = 0 case.  If it isn't found, then return 0.

#if 0
  cerr << "Size of d_data = " << d_data.size() << endl;
  map<string,BoundCondBase*>::const_iterator it;   
  for (unsigned int i = 0; i < d_data.size(); i++) {
    for (it = d_data[i].begin(); it != d_data[i].end(); it++) {
      cerr << "mat id: " << i << " BC " << it->first << endl;
    }
  }
#endif

  if ((int)d_data.size() == 0)
    return 0;

  if ((int)d_data.size() <= mat_id ) {
#if 0
    cerr << "Size of d_data is less than mat_id, must be an ALL" << endl;
#endif
    map<string,Handle<BoundCondBase> >::const_iterator it1;
    it1 = d_data[0].find(type);
    if (it1 == d_data[0].end())
      return 0;
    else
      return it1->second.get_rep();
    
  } else {
    map<string,Handle<BoundCondBase> >::const_iterator iter;   
    iter = d_data[mat_id].find(type);
    if (iter == d_data[mat_id].end()) {
#if 0
      cerr << "Can't find " << type << " for material " << mat_id << endl;
#endif
      iter = d_data[0].find(type);
      if (iter == d_data[0].end()) {
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


