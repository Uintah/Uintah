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


#ifndef Datatypes_PointCloudField_h
#define Datatypes_PointCloudField_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;

template <class Data>
class PointCloudField: public GenericField< PointCloudMesh, vector<Data> >
{ 
public:

  PointCloudField();
  PointCloudField(Field::data_location data_at);
  PointCloudField(PointCloudMeshHandle mesh, Field::data_location data_at);  
  virtual PointCloudField<Data> *clone() const; 
  virtual ~PointCloudField();

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  static Persistent* maker();
};

const int POINT_CLOUD_FIELD_VERSION = 1;

template <class Data>
Persistent* 
PointCloudField<Data>::maker()
{
  return scinew PointCloudField<Data>;
}

template <class Data>
PersistentTypeID
PointCloudField<Data>::type_id(type_name(-1),
		GenericField<PointCloudMesh, vector<Data> >::type_name(-1),
                maker); 

template <class Data>
void
PointCloudField<Data>::io(Piostream &stream)
{
  /*int version=*/stream.begin_class(type_name(-1), POINT_CLOUD_FIELD_VERSION);
  GenericField<PointCloudMesh, vector<Data> >::io(stream);
  stream.end_class();                                                         
}

template <class Data>
PointCloudField<Data>::PointCloudField()
  :  GenericField<PointCloudMesh, vector<Data> >()
{
}


template <class Data>
PointCloudField<Data>::PointCloudField(Field::data_location data_at)
  : GenericField<PointCloudMesh, vector<Data> >(data_at)
{
}


template <class Data>
PointCloudField<Data>::PointCloudField(PointCloudMeshHandle mesh,
			     Field::data_location data_at)
  : GenericField<PointCloudMesh, vector<Data> >(mesh, data_at)
{
}
  

template <class Data>
PointCloudField<Data>::~PointCloudField()
{
}


template <class Data>
PointCloudField<Data> *
PointCloudField<Data>::clone() const 
{
  return new PointCloudField<Data>(*this);
}

 
template <class Data>
const string
PointCloudField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "PointCloudField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class T> 
const TypeDescription*
PointCloudField<T>::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("SCIRun");
  static string path(__FILE__);

  if(!td){
    if (n == -1) {
      const TypeDescription *sub = SCIRun::get_type_description((T*)0);
      TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
      (*subs)[0] = sub;
      td = scinew TypeDescription(name, subs, path, namesp);
    }
    else if(n == 0) {
      td = scinew TypeDescription(name, 0, path, namesp);
    }
    else {
      td = (TypeDescription *) SCIRun::get_type_description((T*)0);
    }
  }
  return td;
}

} // end namespace SCIRun

#endif // Datatypes_PointCloudField_h
