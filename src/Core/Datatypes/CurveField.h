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


#ifndef Datatypes_CurveField_h
#define Datatypes_CurveField_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <string>
#include <vector>

namespace SCIRun {

using std::string;

template <class Data>
class CurveField: public GenericField< CurveMesh, vector<Data> >
{
public:

  CurveField();
  CurveField(Field::data_location data_at);
  CurveField(CurveMeshHandle mesh, Field::data_location data_at);
  virtual CurveField<Data> *clone() const;
  virtual ~CurveField();

  virtual ScalarFieldInterface* query_scalar_interface() const;
  virtual VectorFieldInterface* query_vector_interface() const;
  virtual TensorFieldInterface* query_tensor_interface() const;

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
  virtual const TypeDescription* get_type_description() const;

private:
  static Persistent* maker();
};

#define CURVE_FIELD_VERSION 1

template <class Data>
Persistent*
CurveField<Data>::maker()
{
  return scinew CurveField<Data>;
}

template <class Data>
PersistentTypeID
CurveField<Data>::type_id(type_name(-1),
		GenericField<CurveMesh, vector<Data> >::type_name(-1),
                maker);

template <class Data>
void
CurveField<Data>::io(Piostream &stream)
{
  /*int version=*/stream.begin_class(type_name(-1), CURVE_FIELD_VERSION);
  GenericField<CurveMesh, vector<Data> >::io(stream);
  stream.end_class();
}


template <class Data>
const string
CurveField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "CurveField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
}


template <class Data>
CurveField<Data>::CurveField()
  : GenericField<CurveMesh, vector<Data> >()
{
}


template <class Data>
CurveField<Data>::CurveField(Field::data_location data_at)
  : GenericField<CurveMesh, vector<Data> >(data_at)
{
}


template <class Data>
CurveField<Data>::CurveField(CurveMeshHandle mesh,
				 Field::data_location data_at)
  : GenericField<CurveMesh, vector<Data> >(mesh, data_at)
{
}

template <class Data>
CurveField<Data>::~CurveField()
{
}

template <class Data>
CurveField<Data> *
CurveField<Data>::clone() const
{
  return new CurveField<Data>(*this);
}

template <> ScalarFieldInterface *
CurveField<double>::query_scalar_interface() const;

template <> ScalarFieldInterface *
CurveField<float>::query_scalar_interface() const;

template <> ScalarFieldInterface *
CurveField<int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
CurveField<short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
CurveField<char>::query_scalar_interface() const;

template <> ScalarFieldInterface *
CurveField<unsigned int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
CurveField<unsigned short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
CurveField<unsigned char>::query_scalar_interface() const;

template <class T>
ScalarFieldInterface*
CurveField<T>::query_scalar_interface() const 
{
  return 0;
}

template <>
VectorFieldInterface*
CurveField<Vector>::query_vector_interface() const;

template <class T>
VectorFieldInterface*
CurveField<T>::query_vector_interface() const
{
  return 0;
}

template <>
TensorFieldInterface*
CurveField<Tensor>::query_tensor_interface() const;

template <class T>
TensorFieldInterface*
CurveField<T>::query_tensor_interface() const
{
  return 0;
}

template <class Data>
const string
CurveField<Data>::get_type_name(int n = -1) const
{
  return type_name(n);
}

template <class T>
const TypeDescription* 
get_type_description(CurveField<T>*)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("CurveField", subs, __FILE__, "SCIRun");
  }
  return td;
}

template <class T>
const TypeDescription* 
CurveField<T>::get_type_description() const 
{
  return SCIRun::get_type_description((CurveField<T>*)0);
}

} // end namespace SCIRun

#endif // Datatypes_CurveField_h
