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
 *  TriSurf.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_TriSurf_h
#define Datatypes_TriSurf_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Util/Assert.h>
#include <vector>


namespace SCIRun {

template <class T> 
class TriSurf : public GenericField<TriSurfMesh, vector<T> >
{
public:
  TriSurf();
  TriSurf(Field::data_location data_at);
  TriSurf(TriSurfMeshHandle mesh, Field::data_location data_at);
  virtual TriSurf<T> *clone() const;
  virtual ~TriSurf();
  
  virtual ScalarFieldInterface* query_scalar_interface() const;
  virtual VectorFieldInterface* query_vector_interface() const;

  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  virtual const TypeDescription* get_type_description() const;

private:
  static Persistent *maker();
};

// Pio defs.
const int TRI_SURF_VERSION = 1;

template <class T>
Persistent *
TriSurf<T>::maker()
{
  return scinew TriSurf<T>;
}

template <class T>
PersistentTypeID 
TriSurf<T>::type_id(type_name(-1), 
		    GenericField<TriSurfMesh, vector<T> >::type_name(-1),
		    maker);


template <class T>
void 
TriSurf<T>::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), TRI_SURF_VERSION);
  GenericField<TriSurfMesh, vector<T> >::io(stream);
  stream.end_class();
}


template <class T> 
const string 
TriSurf<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "TriSurf";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T>
TriSurf<T>::TriSurf()
  : GenericField<TriSurfMesh, vector<T> >()
{
}

template <class T>
TriSurf<T>::TriSurf(Field::data_location data_at) :
  GenericField<TriSurfMesh, vector<T> >(data_at)
{
}

template <class T>
TriSurf<T>::TriSurf(TriSurfMeshHandle mesh, Field::data_location data_at)
  : GenericField<TriSurfMesh, vector<T> >(mesh, data_at)
{
} 

template <class T>
TriSurf<T> *
TriSurf<T>::clone() const
{
  return new TriSurf(*this);
}

template <class T>
TriSurf<T>::~TriSurf()
{
}

template <> ScalarFieldInterface *
TriSurf<double>::query_scalar_interface() const;

template <> ScalarFieldInterface *
TriSurf<float>::query_scalar_interface() const;

template <> ScalarFieldInterface *
TriSurf<int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
TriSurf<short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
TriSurf<unsigned char>::query_scalar_interface() const;

template <class T>
ScalarFieldInterface*
TriSurf<T>::query_scalar_interface() const 
{
  return 0;
}

template <>
VectorFieldInterface*
TriSurf<Vector>::query_vector_interface() const;

template <class T>
VectorFieldInterface*
TriSurf<T>::query_vector_interface() const
{
  return 0;
}


template <class T>
const string 
TriSurf<T>::get_type_name(int n = -1) const
{
  return type_name(n);
}

template <class T>
const TypeDescription* 
get_type_description(TriSurf<T>*)
{
  static TypeDescription* td = 0;
  static string name("TriSurf");
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
TriSurf<T>::get_type_description() const 
{
  return SCIRun::get_type_description((TriSurf<T>*)0);
}

} // end namespace SCIRun

#endif // Datatypes_TriSurf_h



















