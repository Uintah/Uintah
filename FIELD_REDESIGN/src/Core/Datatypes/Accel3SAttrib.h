//  Accel3SAttrib.h - scalar attribute stored as a flat array
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_Accel3SAttrib_h
#define SCI_project_Accel3SAttrib_h 1

#include <SCICore/Datatypes/FlatSAttrib.h>

namespace SCICore {
namespace Datatypes {

using std::vector;
using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Exceptions::ArrayIndexOutOfBounds;
using SCICore::Exceptions::DimensionMismatch;
using SCICore::Math::Min;
using SCICore::Math::Max;
using std::ostringstream;
using SCICore::Util::DebugStream;

template <class T> class Accel3SAttrib:public FlatSAttrib<T>
{
public:
  
  //////////
  // Constructors
  Accel3SAttrib();
  Accel3SAttrib(const FlatSAttrib<T> & copy);
  Accel3SAttrib(int, int, int);

  //////////
  // Destructor
  ~Accel3SAttrib();
  
  //////////
  // return the value at the given position
  T& grid(int, int, int);
  T& grid(int, int);
  T& grid(int);

  //////////
  // Resize the attribute to the specified dimensions
  virtual void resize(int, int, int);
  virtual void resize(int, int);
  virtual void resize(int);

  virtual string get_info();  

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  
private:
  void create_accel_structure();

  // Accel structure.
  vector<vector<T *> > accel;
  static DebugStream dbg;
};

}
}

namespace SCICore {
namespace Datatypes {

template <class T> DebugStream Accel3SAttrib<T>::dbg("Accel3SAttrib", true);

template <class T> PersistentTypeID Accel3SAttrib<T>::type_id("Accel3SAttrib", "Datatype", 0);

template <class T> Accel3SAttrib<T>::Accel3SAttrib()
  :FlatSAttrib<T>() {
}

template <class T> Accel3SAttrib<T>::Accel3SAttrib(const FlatSAttrib<T>& copy)
  : FlatSAttrib<T>(copy) {
  create_accel_structure();
}

template <class T> Accel3SAttrib<T>::Accel3SAttrib(int ix, int iy, int iz):
  FlatSAttrib<T>(ix, iy, iz){
  create_accel_structure();
}

template <class T> Accel3SAttrib<T>::~Accel3SAttrib() {
}


template <class T> T& Accel3SAttrib<T>::grid(int ix, int iy, int iz) {
  return accel[iz][iy][ix];
}

template <class T> T& Accel3SAttrib<T>::grid(int ix, int iy) {
  throw DimensionMismatch(2, 3);
  return accel[0][iy][ix];
}


template <class T> T& Accel3SAttrib<T>::grid(int ix) {
  throw DimensionMismatch(1, 3);
  return accel[0][0][ix];
}


template <class T> void Accel3SAttrib<T>::resize(int x, int y, int z) {
  FlatSAttrib<T>::resize(x, y, z);
  create_accel_structure();
}


template <class T> void Accel3SAttrib<T>::resize(int x, int y) {
  throw DimensionMismatch(2, 3);
}


template <class T> void Accel3SAttrib<T>::resize(int x) {
  throw DimensionMismatch(1, 3);
}


template <class T> void Accel3SAttrib<T>::create_accel_structure() {
  accel.resize(nz);
  for (int i=0; i < nz; i++) {
    accel[i].resize(ny);
    for (int j=0; j < ny; j++) {
      accel[i][j] = &data[i*nx*ny + j*nx];
    }
  }
}


template <class T> string Accel3SAttrib<T>::get_info() {
  ostringstream retval;
  retval <<
    "Name = " << name << '\n' <<
    "Type = Accel3SAttrib" << '\n' <<
    "Size = " << data.size() << '\n' <<
    "Data = ";
  vector<T>::iterator itr = data.begin();
  for(;itr!=data.end(); itr++) {
    retval << *itr << " ";
  }
  retval << endl;
  return retval.str();
}

template <class T> void Accel3SAttrib<T>::io(Piostream&){
}

}  // end Datatypes
}  // end SCICore



#endif  // SCI_project_Accel3SAttrib_h



