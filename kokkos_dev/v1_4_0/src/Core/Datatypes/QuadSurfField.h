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

/*
 *  QuadSurfField.h
 *
 *  Written by:
 *   Michael Callahan
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_QuadSurfField_h
#define Datatypes_QuadSurfField_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Util/Assert.h>
#include <vector>


namespace SCIRun {

template <class T> 
class QuadSurfField : public GenericField<QuadSurfMesh, vector<T> >
{
public:
  QuadSurfField();
  QuadSurfField(Field::data_location data_at);
  QuadSurfField(QuadSurfMeshHandle mesh, Field::data_location data_at);
  virtual QuadSurfField<T> *clone() const;
  virtual ~QuadSurfField();
  
  virtual ScalarFieldInterface* query_scalar_interface() const;
  virtual VectorFieldInterface* query_vector_interface() const;
  virtual TensorFieldInterface* query_tensor_interface() const;

  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  virtual const TypeDescription* get_type_description() const;

private:
  static Persistent *maker();
};

// Pio defs.
const int QUAD_SURF_FIELD_VERSION = 1;

template <class T>
Persistent *
QuadSurfField<T>::maker()
{
  return scinew QuadSurfField<T>;
}

template <class T>
PersistentTypeID 
QuadSurfField<T>::type_id(type_name(-1), 
		    GenericField<QuadSurfMesh, vector<T> >::type_name(-1),
		    maker);


template <class T>
void 
QuadSurfField<T>::io(Piostream& stream)
{
  /*int version=*/stream.begin_class(type_name(-1), QUAD_SURF_FIELD_VERSION);
  GenericField<QuadSurfMesh, vector<T> >::io(stream);
  stream.end_class();
}


template <class T> 
const string 
QuadSurfField<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "QuadSurfField";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T>
QuadSurfField<T>::QuadSurfField()
  : GenericField<QuadSurfMesh, vector<T> >()
{
}

template <class T>
QuadSurfField<T>::QuadSurfField(Field::data_location data_at) :
  GenericField<QuadSurfMesh, vector<T> >(data_at)
{
}

template <class T>
QuadSurfField<T>::QuadSurfField(QuadSurfMeshHandle mesh, Field::data_location data_at)
  : GenericField<QuadSurfMesh, vector<T> >(mesh, data_at)
{
} 

template <class T>
QuadSurfField<T> *
QuadSurfField<T>::clone() const
{
  return new QuadSurfField(*this);
}

template <class T>
QuadSurfField<T>::~QuadSurfField()
{
}

template <> ScalarFieldInterface *
QuadSurfField<double>::query_scalar_interface() const;

template <> ScalarFieldInterface *
QuadSurfField<float>::query_scalar_interface() const;

template <> ScalarFieldInterface *
QuadSurfField<int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
QuadSurfField<short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
QuadSurfField<char>::query_scalar_interface() const;

template <> ScalarFieldInterface *
QuadSurfField<unsigned int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
QuadSurfField<unsigned short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
QuadSurfField<unsigned char>::query_scalar_interface() const;

template <class T>
ScalarFieldInterface*
QuadSurfField<T>::query_scalar_interface() const 
{
  return 0;
}

template <>
VectorFieldInterface*
QuadSurfField<Vector>::query_vector_interface() const;

template <class T>
VectorFieldInterface*
QuadSurfField<T>::query_vector_interface() const
{
  return 0;
}

template <>
TensorFieldInterface*
QuadSurfField<Tensor>::query_tensor_interface() const;

template <class T>
TensorFieldInterface*
QuadSurfField<T>::query_tensor_interface() const
{
  return 0;
}


template <class T>
const string 
QuadSurfField<T>::get_type_name(int n = -1) const
{
  return type_name(n);
}

template <class T>
const TypeDescription* 
get_type_description(QuadSurfField<T>*)
{
  static TypeDescription* td = 0;
  static string name("QuadSurfField");
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
QuadSurfField<T>::get_type_description() const 
{
  return SCIRun::get_type_description((QuadSurfField<T>*)0);
}

} // end namespace SCIRun

#endif // Datatypes_QuadSurfField_h



















