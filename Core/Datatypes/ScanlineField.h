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


#ifndef Datatypes_ScanlineField_h
#define Datatypes_ScanlineField_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ScanlineMesh.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array2.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <string>

namespace SCIRun {

using std::string;


template <class Data>
class ScanlineField : public GenericField< ScanlineMesh, vector<Data> >
{
public:
  ScanlineField();
  ScanlineField(Field::data_location data_at);
  ScanlineField(ScanlineMeshHandle mesh, Field::data_location data_at);
  virtual ScanlineField<Data> *clone() const;
  virtual ~ScanlineField();

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



template <class Data>
ScanlineField<Data>::ScanlineField()
  : GenericField<ScanlineMesh, vector<Data> >()
{
}


template <class Data>
ScanlineField<Data>::ScanlineField(Field::data_location data_at)
  : GenericField<ScanlineMesh, vector<Data> >(data_at)
{
}


template <class Data>
ScanlineField<Data>::ScanlineField(ScanlineMeshHandle mesh,
			     Field::data_location data_at)
  : GenericField<ScanlineMesh, vector<Data> >(mesh, data_at)
{
}


template <class Data>
ScanlineField<Data> *
ScanlineField<Data>::clone() const
{
  return new ScanlineField(*this);
}
  

template <class Data>
ScanlineField<Data>::~ScanlineField()
{
}

template <> ScalarFieldInterface *
ScanlineField<double>::query_scalar_interface() const;

template <> ScalarFieldInterface *
ScanlineField<int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
ScanlineField<short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
ScanlineField<unsigned char>::query_scalar_interface() const;

template <class T>
ScalarFieldInterface*
ScanlineField<T>::query_scalar_interface() const 
{
  return 0;
}

template <>
VectorFieldInterface*
ScanlineField<Vector>::query_vector_interface() const;

template <class T>
VectorFieldInterface*
ScanlineField<T>::query_vector_interface() const
{
  return 0;
}


template <class Data>
const string
ScanlineField<Data>::get_type_name(int n) const
{
  return type_name(n);
}


#define SCANLINEFIELD_VERSION 1

template <class Data>
Persistent* 
ScanlineField<Data>::maker()
{
  return scinew ScanlineField<Data>;
}

template <class Data>
PersistentTypeID
ScanlineField<Data>::type_id(type_name(-1),
		GenericField<ScanlineMesh, vector<Data> >::type_name(-1),
                maker); 

template <class Data>
void
ScanlineField<Data>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), SCANLINEFIELD_VERSION);
  GenericField<ScanlineMesh, vector<Data> >::io(stream);
  stream.end_class();                                                         
}


template <class Data>
const string
ScanlineField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "ScanlineField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class T>
const TypeDescription* 
get_type_description(ScanlineField<T>*)
{
  static TypeDescription* td = 0;
  static string name("ScanlineField");  
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
ScanlineField<T>::get_type_description() const 
{
  return SCIRun::get_type_description((ScanlineField<T>*)0);
}


} // end namespace SCIRun

#endif // Datatypes_ScanlineField_h
