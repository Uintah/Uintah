/*
 *  TetVol.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_TetVol_h
#define Datatypes_TetVol_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <vector>


namespace SCIRun {

template <class T> 
class TetVol : public GenericField<TetVolMesh, vector<T> > {
public:
  GenericField() {};
  GenericField(data_location data_at) {};
  virtual ~GenericField() {};
 
};

// FIX_ME support the int arg return vals...
template <class T> 
TetVol::type_name(int a)
{
  const static string name =  "TetVol<" + find_type_name((T *)0) + ">";
  return name;
}

} // end namespace SCIRun

#endif // Datatypes_TetVol_h



















