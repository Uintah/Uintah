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


#ifndef Datatypes_ContourField_h
#define Datatypes_ContourField_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ContourMesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <string>
#include <vector>

namespace SCIRun {

using std::string;

template <class Data>
class ContourField: public GenericField< ContourMesh, vector<Data> >
{
public:

  ContourField();
  ContourField(Field::data_location data_at);
  ContourField(ContourMeshHandle mesh, Field::data_location data_at);
  virtual ContourField<Data> *clone() const;
  virtual ~ContourField();

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

#define CONTOURFIELD_VERSION 1

template <class Data>
Persistent*
ContourField<Data>::maker()
{
  return scinew ContourField<Data>;
}

template <class Data>
PersistentTypeID
ContourField<Data>::type_id(type_name(-1),
		GenericField<ContourMesh, vector<Data> >::type_name(-1),
                maker);

template <class Data>
void
ContourField<Data>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), CONTOURFIELD_VERSION);
  GenericField<ContourMesh, vector<Data> >::io(stream);
  stream.end_class();
}


template <class Data>
const string
ContourField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "ContourField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
}


template <class Data>
ContourField<Data>::ContourField()
  : GenericField<ContourMesh, vector<Data> >()
{
}


template <class Data>
ContourField<Data>::ContourField(Field::data_location data_at)
  : GenericField<ContourMesh, vector<Data> >(data_at)
{
}


template <class Data>
ContourField<Data>::ContourField(ContourMeshHandle mesh,
				 Field::data_location data_at)
  : GenericField<ContourMesh, vector<Data> >(mesh, data_at)
{
}

template <class Data>
ContourField<Data>::~ContourField()
{
}

template <class Data>
ContourField<Data> *
ContourField<Data>::clone() const
{
  return new ContourField<Data>(*this);
}

template <> ScalarFieldInterface *
ContourField<double>::query_scalar_interface() const;

template <> ScalarFieldInterface *
ContourField<int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
ContourField<short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
ContourField<unsigned char>::query_scalar_interface() const;

template <class T>
ScalarFieldInterface*
ContourField<T>::query_scalar_interface() const 
{
  return 0;
}

template <>
VectorFieldInterface*
ContourField<Vector>::query_vector_interface() const;

template <class T>
VectorFieldInterface*
ContourField<T>::query_vector_interface() const
{
  return 0;
}

template <class Data>
const string
ContourField<Data>::get_type_name(int n = -1) const
{
  return type_name(n);
}

template <class T>
const TypeDescription* 
get_type_description(ContourField<T>*)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("ContourField", subs, __FILE__, "SCIRun");
  }
  return td;
}

template <class T>
const TypeDescription* 
ContourField<T>::get_type_description() const 
{
  return SCIRun::get_type_description((ContourField<T>*)0);
}

} // end namespace SCIRun

#endif // Datatypes_ContourField_h
