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
#include <vector>


namespace SCIRun {

template <class T> 
class TriSurf : public GenericField<TriSurfMesh, vector<T> > {
public:
  TriSurf() : 
    GenericField<TriSurfMesh, vector<T> >() {};
  TriSurf(Field::data_location data_at) : 
    GenericField<TriSurfMesh, vector<T> >(data_at) {};
  virtual ~TriSurf() {};

  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  //static const string type_name(int a);
  static const string type_name();
  static const string type_name(int);
 
private:
  static Persistent *maker();
};

// Pio defs.
const double TET_VOL_VERSION = 1.0;


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
  stream.begin_class(type_name().c_str(), TET_VOL_VERSION);
  GenericField<TriSurfMesh, vector<T> >::io(stream);
  stream.end_class();
}

// FIX_ME support the int arg return vals...
template <class T> 
const string 
TriSurf<T>::type_name()
{
  static const string name =  "TriSurf<" + find_type_name((T *)0) + ">";
  return name;
}

template <class T> 
const string 
TriSurf<T>::type_name(int a)
{
  ASSERT((a <= 1) && a >= 0);
  if (a == 0) { return "TriSurf"; }
  return find_type_name((T *)0);
}

#if defined(__sgi)  
#pragma reset woff 1424
#endif

} // end namespace SCIRun

#endif // Datatypes_TriSurf_h



















