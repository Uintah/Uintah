#include <Packages/Uintah/Core/Grid/BCData.h>
#include <Packages/Uintah/Core/Grid/BoundCondBase.h>

using namespace Uintah;
#include <iostream>
#include <algorithm>
using std::cerr;
using std::endl;

#define PRINT
#undef PRINT

BCData::BCData() 
{
}

BCData::~BCData()
{
   bcDataType::const_iterator itr;
  for (itr = d_BCData.begin(); itr != d_BCData.end(); itr++) {
     delete itr->second;
  }
  d_BCData.clear();
}

BCData::BCData(const BCData& rhs)
{
  bcDataType* m = const_cast<bcDataType *>(&(rhs.d_BCData));
  bcDataType::const_iterator itr;
  for (itr = m->begin(); itr != m->end(); itr++)
    d_BCData[itr->first]=itr->second->clone();
}

BCData& BCData::operator=(const BCData& rhs)
{
  if (this == &rhs)
    return *this;

  // Delete the lhs
  bcDataType::const_iterator itr;
  for (itr = d_BCData.begin(); itr != d_BCData.end(); ++itr) 
    delete itr->second;

  d_BCData.clear();
  // Copy the rhs to the lhs
  bcDataType* m = const_cast<bcDataType *>(&(rhs.d_BCData));
  for (itr = m->begin(); itr != m->end(); itr++)
    d_BCData[itr->first] = itr->second->clone();
  
  return *this;
}

bool BCData::operator==(const BCData& rhs)
{
  bcDataType::const_iterator itr;
  for (itr = d_BCData.begin(); itr != d_BCData.end();) {
    if (rhs.find(itr->first) == false)
      return false;
    else
      ++itr;
  }
  return true;
}

bool BCData::operator<(const BCData& rhs) const
{
  bcDataType::const_iterator itr;
  if (d_BCData.size() < rhs.d_BCData.size())
    return true;
  else 
    return false;
}

void 
BCData::setBCValues(BoundCondBase* bc)
{
  d_BCData[bc->getType()]=bc->clone();
}



const BoundCondBase*
BCData::getBCValues(const string& type) const
{
  // The default location for BCs defined for all materials is mat_id = -1.
  // Need to first check the actual mat_id specified.  If this is not found,
  // then will check mat_id = -1 case.  If it isn't found, then return 0.


  bcDataType::const_iterator itr;
#ifdef PRINT
  cerr << "Finding " << type << endl;
#endif
  itr = d_BCData.find(type);
  if (itr != d_BCData.end()) {
#ifdef PRINT
    cerr << "Found it! " << endl;
    cerr << "bctype return = " << typeid(*(itr->second)).name() << endl;
#endif
    return itr->second->clone();
  }  else {
#ifdef PRINT
    cerr << "Didn't find it" << endl;
#endif
    return 0;
  }
}

bool BCData::find(const string& type) const
{
  bcDataType:: const_iterator itr;

  itr = d_BCData.find(type);
  if (itr != d_BCData.end())
    return true;
  else
    return false;
}


void BCData::print() const
{
  bcDataType::const_iterator itr;
  for (itr = d_BCData.begin(); itr != d_BCData.end(); itr++) {
    cerr << "BC = " << itr->first << " actual name = " 
	 << itr->second->getType() << " bctype = " 
	 << typeid(*(itr->second)).name() << " " << itr->second << endl;
  }
  
}
