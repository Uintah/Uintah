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
 *  TriSurfField.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_TriSurfField_h
#define Datatypes_TriSurfField_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Util/Assert.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>


namespace SCIRun {
using std::vector;

template <class T> 
class TriSurfField : public GenericField<TriSurfMesh, vector<T> >
{
public:
  TriSurfField();
  TriSurfField(Field::data_location data_at);
  TriSurfField(TriSurfMeshHandle mesh, Field::data_location data_at);
  virtual TriSurfField<T> *clone() const;
  virtual ~TriSurfField();
  
  //! Persistent IO
  static PersistentTypeID type_id;
  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;
  virtual void io(Piostream &stream);

private:
  static Persistent *maker();
};

// Pio defs.
const int TRI_SURF_FIELD_VERSION = 1;

template <class T>
Persistent *
TriSurfField<T>::maker()
{
  return scinew TriSurfField<T>;
}

template <class T>
PersistentTypeID 
TriSurfField<T>::type_id(type_name(-1), 
		    GenericField<TriSurfMesh, vector<T> >::type_name(-1),
		    maker);


template <class T>
void 
TriSurfField<T>::io(Piostream& stream)
{
  /*int version=*/stream.begin_class(type_name(-1), TRI_SURF_FIELD_VERSION);
  GenericField<TriSurfMesh, vector<T> >::io(stream);
  stream.end_class();
}

template <class T> 
const string 
TriSurfField<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "TriSurfField";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T> 
const TypeDescription*
TriSurfField<T>::get_type_description(int n) const
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

template <class T>
TriSurfField<T>::TriSurfField()
  : GenericField<TriSurfMesh, vector<T> >()
{
}

template <class T>
TriSurfField<T>::TriSurfField(Field::data_location data_at) :
  GenericField<TriSurfMesh, vector<T> >(data_at)
{
}

template <class T>
TriSurfField<T>::TriSurfField(TriSurfMeshHandle mesh, Field::data_location data_at)
  : GenericField<TriSurfMesh, vector<T> >(mesh, data_at)
{
} 

template <class T>
TriSurfField<T> *
TriSurfField<T>::clone() const
{
  return new TriSurfField(*this);
}

template <class T>
TriSurfField<T>::~TriSurfField()
{
}

} // end namespace SCIRun

#endif // Datatypes_TriSurfField_h



















