//  Accel2SAttrib.h - scalar attribute stored as a flat array
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_Accel2SAttrib_h
#define SCI_project_Accel2SAttrib_h 1

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

template <class T> class Accel2SAttrib:public FlatSAttrib<T>
{
public:
  
  //////////
  // Constructors
  Accel2SAttrib();
  Accel2SAttrib(const FlatSAttrib<T>& copy);
  Accel2SAttrib(int, int);

  //////////
  // Destructor
  ~Accel2SAttrib();
  
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
  vector<T *> accel;
  static DebugStream dbg;
};

}
}

namespace SCICore {
namespace Datatypes {

template <class T> DebugStream Accel2SAttrib<T>::dbg("Accel2SAttrib", true);

template <class T> PersistentTypeID Accel2SAttrib<T>::type_id("Accel2SAttrib", "Datatype", 0);

template <class T> Accel2SAttrib<T>::Accel2SAttrib()
  :FlatSAttrib<T>() {
}

template <class T> Accel2SAttrib<T>::Accel2SAttrib(const FlatSAttrib<T>& copy)
  : FlatSAttrib<T>(copy) {
  create_accel_structure();
}

template <class T> Accel2SAttrib<T>::Accel2SAttrib(int ix, int iy):
  FlatSAttrib<T>(ix, iy) {
  create_accel_structure();
}

template <class T> Accel2SAttrib<T>::~Accel2SAttrib(){
}


template <class T> T& Accel2SAttrib<T>::grid(int ix, int iy, int iz) {
  throw DimensionMismatch(3, 2);
  return accel[iy][ix];
}

template <class T> T& Accel2SAttrib<T>::grid(int ix, int iy) {
  return accel[iy][ix];
}

template <class T> T& Accel2SAttrib<T>::grid(int ix) {
  throw DimensionMismatch(1, 2);
  return accel[0][ix];
}

template <class T> void Accel2SAttrib<T>::resize(int x, int y, int z) {
  throw DimensionMismatch(3, 2);
}

template <class T> void Accel2SAttrib<T>::resize(int x, int y) {
  FlatSAttrib<T>::resize(x, y);
  create_accel_structure();
}

template <class T> void Accel2SAttrib<T>::resize(int x) {
  throw DimensionMismatch(1, 2);
}


template <class T> void Accel2SAttrib<T>::create_accel_structure() {
  accel.resize(ny);
  for (int i=0; i < ny; i++)
    {
      accel[i] = &data[i * nx];
    }
}
  

template <class T> string Accel2SAttrib<T>::get_info() {
  ostringstream retval;
  retval <<
    "Name = " << name << '\n' <<
    "Type = Accel2SAttrib" << '\n' <<
    "Size = " << data.size() << '\n' <<
    "Data = ";
  vector<T>::iterator itr = data.begin();
  for(;itr!=data.end(); itr++) {
    retval << *itr << " ";
  }
  retval << endl;
  return retval.str();
}


template <class T> void Accel2SAttrib<T>::io(Piostream&){
}

}  // end Datatypes
}  // end SCICore



#endif  // SCI_project_Accel2SAttrib_h



