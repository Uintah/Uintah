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
 *  StructQuadSurfField.cc: Templated Field defined on a 2D Structured Grid
 *
 *  Written by:
 *   Allen Sanderson
 *   Department of Computer Science
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 2002 SCI Group
 *
 */

/*
  See StructQuadSurfMesh.h for field/mesh comments.
*/

#ifndef Datatypes_StructQuadSurfField_h
#define Datatypes_StructQuadSurfField_h

#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Util/Assert.h>

namespace SCIRun {

template <class Data>
class StructQuadSurfField : public GenericField< StructQuadSurfMesh, FData2d<Data> >
{
public:
  StructQuadSurfField();
  StructQuadSurfField(Field::data_location data_at);
  StructQuadSurfField(StructQuadSurfMeshHandle mesh, Field::data_location data_at);
  virtual StructQuadSurfField<Data> *clone() const;
  virtual ~StructQuadSurfField();

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  static Persistent* maker();
};



template <class Data>
StructQuadSurfField<Data>::StructQuadSurfField()
  : GenericField<StructQuadSurfMesh, FData2d<Data> >()
{
}


template <class Data>
StructQuadSurfField<Data>::StructQuadSurfField(Field::data_location data_at)
  : GenericField<StructQuadSurfMesh, FData2d<Data> >(data_at)
{
}


template <class Data>
StructQuadSurfField<Data>::StructQuadSurfField(StructQuadSurfMeshHandle mesh,
			     Field::data_location data_at)
  : GenericField<StructQuadSurfMesh, FData2d<Data> >(mesh, data_at)
{
}


template <class Data>
StructQuadSurfField<Data> *
StructQuadSurfField<Data>::clone() const
{
  return new StructQuadSurfField(*this);
}
  

template <class Data>
StructQuadSurfField<Data>::~StructQuadSurfField()
{
}


#define STRUCT_QUAD_SURF_FIELD_VERSION 1

template <class Data>
Persistent* 
StructQuadSurfField<Data>::maker()
{
  return scinew StructQuadSurfField<Data>;
}

template <class Data>
PersistentTypeID
StructQuadSurfField<Data>::type_id(type_name(-1),
		GenericField<StructQuadSurfMesh, FData2d<Data> >::type_name(-1),
                maker); 

template <class Data>
void
StructQuadSurfField<Data>::io(Piostream &stream)
{
  int version = stream.begin_class(type_name(-1), STRUCT_QUAD_SURF_FIELD_VERSION);
  GenericField<StructQuadSurfMesh, FData2d<Data> >::io(stream);
  stream.end_class();                                                         
  if (version < 2) {
    FData2d<Data> temp;
    temp.copy(fdata());
    resize_fdata();
    int i, j;
    for (i=0; i<fdata().dim1(); i++)
      for (j=0; j<fdata().dim2(); j++)
	fdata()(i,j)=temp(j,i);
  }  
}

template <class Data>
const string
StructQuadSurfField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "StructQuadSurfField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class T> 
const TypeDescription*
StructQuadSurfField<T>::get_type_description(int n) const
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

#endif // Datatypes_StructQuadSurfField_h
