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
class TriSurf : public GenericField<TriSurfMesh, vector<T> > {
public:
  TriSurf() : 
    GenericField<TriSurfMesh, vector<T> >() {}
  TriSurf(Field::data_location data_at) : 
    GenericField<TriSurfMesh, vector<T> >(data_at) {}
  TriSurf(TriSurfMeshHandle mesh, Field::data_location data_at) : 
    GenericField<TriSurfMesh, vector<T> >(mesh, data_at) {} 
  virtual Field *clone() { return new TriSurf(*this); }
  virtual ~TriSurf() {};
  
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

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
TriSurf<T>::type_id(type_name(), 
		    GenericField<TriSurfMesh, vector<T> >::type_name(),
		    maker);


template <class T>
void 
TriSurf<T>::io(Piostream& stream)
{
  stream.begin_class(type_name().c_str(), TRI_SURF_VERSION);
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


} // end namespace SCIRun

#endif // Datatypes_TriSurf_h



















