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
#include <Core/Util/Assert.h>
#include <vector>

namespace SCIRun {

template <class T> 
class TetVol : public GenericField<TetVolMesh, vector<T> > {
public:
  TetVol() : 
    GenericField<TetVolMesh, vector<T> >() {};
  TetVol(Field::data_location data_at) : 
    GenericField<TetVolMesh, vector<T> >(data_at) {};
  TetVol(TetVolMeshHandle mesh, Field::data_location data_at) : 
    GenericField<TetVolMesh, vector<T> >(mesh, data_at) 
  {
    resize_fdata();
  };

  virtual ~TetVol() {};

  /*! Ask mesh to compute edges and faces. Does nothing if mesh 
    is already finished. */
  void finish_mesh() { get_typed_mesh()->finish(); }

  //! Persistent IO
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;

  void resize_fdata() {
    if (data_at() == NODE)
      fdata().resize(get_typed_mesh()->nodes_size());
    else if (data_at() == EDGE)
      fdata().resize(get_typed_mesh()->edges_size());
    else if (data_at() == FACE)
      ASSERTFAIL("tetvol can't have data at faces (yet)")
    else if (data_at() == CELL)
      fdata().resize(get_typed_mesh()->cells_size());
    else
      ASSERTFAIL("data at unrecognized location")
  }

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

private:
  static Persistent *maker();
};

// Pio defs.
const int TET_VOL_VERSION = 1;

template <class T>
Persistent*
TetVol<T>::maker()
{
  return scinew TetVol<T>;
}

template <class T>
PersistentTypeID 
TetVol<T>::type_id(type_name(), 
		   GenericField<TetVolMesh, vector<T> >::type_name(),
		   maker);


template <class T>
void 
TetVol<T>::io(Piostream& stream)
{
  stream.begin_class(type_name().c_str(), TET_VOL_VERSION);
  GenericField<TetVolMesh, vector<T> >::io(stream);
  stream.end_class();
}

template <class T> 
const string 
TetVol<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "TetVol";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

} // end namespace SCIRun

#endif // Datatypes_TetVol_h
