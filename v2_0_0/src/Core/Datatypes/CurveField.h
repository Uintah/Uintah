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
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_off.h>

namespace SCIRun {

using std::string;

template <class T>
class CurveField: public GenericField< CurveMesh, vector<T> >
{
public:

  CurveField();
  CurveField(Field::data_location data_at);
  CurveField(CurveMeshHandle mesh, Field::data_location data_at);
  virtual CurveField<T> *clone() const;
  virtual ~CurveField();

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  static Persistent* maker();
};

#define CURVE_FIELD_VERSION 1

template <class T>
Persistent*
CurveField<T>::maker()
{
  return scinew CurveField<T>;
}

template <class T>
PersistentTypeID
CurveField<T>::type_id(type_name(-1),
		GenericField<CurveMesh, vector<T> >::type_name(-1),
                maker);

template <class T>
void
CurveField<T>::io(Piostream &stream)
{
  /*int version=*/stream.begin_class(type_name(-1), CURVE_FIELD_VERSION);
  GenericField<CurveMesh, vector<T> >::io(stream);
  stream.end_class();
}


template <class T>
CurveField<T>::CurveField()
  : GenericField<CurveMesh, vector<T> >()
{
}


template <class T>
CurveField<T>::CurveField(Field::data_location data_at)
  : GenericField<CurveMesh, vector<T> >(data_at)
{
}


template <class T>
CurveField<T>::CurveField(CurveMeshHandle mesh,
				 Field::data_location data_at)
  : GenericField<CurveMesh, vector<T> >(mesh, data_at)
{
}

template <class T>
CurveField<T>::~CurveField()
{
}

template <class T>
CurveField<T> *
CurveField<T>::clone() const
{
  return new CurveField<T>(*this);
}


template <class T>
const string
CurveField<T>::type_name(int n)
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
    return find_type_name((T *)0);
  }
}

template <class T> 
const TypeDescription*
CurveField<T>::get_type_description(int n) const
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

#endif // Datatypes_CurveField_h














