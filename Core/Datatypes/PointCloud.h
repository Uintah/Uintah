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
  virtual PointCloud<Data> *clone() const; 
  virtual ~PointCloud();

  virtual ScalarFieldInterface* query_scalar_interface() const;
  virtual VectorFieldInterface* query_vector_interface() const;
 
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
  virtual const TypeDescription* get_type_description() const;

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
  stream.begin_class(type_name(-1), PointCloud_VERSION);
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
 
template <> ScalarFieldInterface *
PointCloud<double>::query_scalar_interface() const;

template <> ScalarFieldInterface *
PointCloud<int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
PointCloud<short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
PointCloud<unsigned char>::query_scalar_interface() const;

template <class T>
ScalarFieldInterface*
PointCloud<T>::query_scalar_interface() const 
{
  return 0;
}

template <>
VectorFieldInterface*
PointCloud<Vector>::query_vector_interface() const;

template <class T>
VectorFieldInterface*
PointCloud<T>::query_vector_interface() const
{
  return 0;
}

template <class Data>
const string 
PointCloud<Data>::get_type_name(int n = -1) const
{
  return type_name(n);
}

template <class T>
const TypeDescription* 
get_type_description(PointCloud<T>*)
{
  static TypeDescription* td = 0;
  static string name("PointCloud");
  static string namesp("SCIRun");
  static string path(__FILE__);
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(name, subs, path, namesp);
  }
  return td;
}

template <class T>
const TypeDescription* 
PointCloud<T>::get_type_description() const 
{
  return SCIRun::get_type_description((PointCloud<T>*)0);
}

} // end namespace SCIRun

#endif // Datatypes_PointCloud_h
