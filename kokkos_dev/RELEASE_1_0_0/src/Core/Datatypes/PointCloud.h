/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


#ifndef Datatypes_PointCloud_h
#define Datatypes_PointCloud_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <string>
#include <vector>

namespace SCIRun {

using std::string;

template <class Data>
class PointCloud: public GenericField< PointCloudMesh, vector<Data> >
{ 
public:

  PointCloud();
  PointCloud(Field::data_location data_at);
  PointCloud(PointCloudMeshHandle mesh, Field::data_location data_at);  
  virtual ~PointCloud();

  virtual PointCloud<Data> *clone() const; 
 
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
private:
  static Persistent* maker();
};

#define PointCloud_VERSION 1

template <class Data>
Persistent* 
PointCloud<Data>::maker()
{
  return scinew PointCloud<Data>;
}

template <class Data>
PersistentTypeID
PointCloud<Data>::type_id(type_name(-1),
		GenericField<PointCloudMesh, vector<Data> >::type_name(-1),
                maker); 

template <class Data>
void
PointCloud<Data>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1).c_str(), PointCloud_VERSION);
  GenericField<PointCloudMesh, vector<Data> >::io(stream);
  stream.end_class();                                                         
}


template <class Data>
const string
PointCloud<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "PointCloud";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 


template <class Data>
PointCloud<Data>::PointCloud()
  :  GenericField<PointCloudMesh, vector<Data> >()
{
}


template <class Data>
PointCloud<Data>::PointCloud(Field::data_location data_at)
  : GenericField<PointCloudMesh, vector<Data> >(data_at)
{
}


template <class Data>
PointCloud<Data>::PointCloud(PointCloudMeshHandle mesh,
			     Field::data_location data_at)
  : GenericField<PointCloudMesh, vector<Data> >(mesh, data_at)
{
}
  

template <class Data>
PointCloud<Data>::~PointCloud()
{
}


template <class Data>
PointCloud<Data> *
PointCloud<Data>::clone() const 
{
  return new PointCloud<Data>(*this);
}
 

template <class Data>
const string 
PointCloud<Data>::get_type_name(int n = -1) const
{
  return type_name(n);
}


} // end namespace SCIRun

#endif // Datatypes_PointCloud_h
















