//  FlatAttrib.h - scalar attribute stored as a flat array
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_FlatAttrib_h
#define SCI_project_FlatAttrib_h 1

#include <vector>

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Datatypes/DiscreteAttrib.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Exceptions/ArrayIndexOutOfBounds.h>
#include <SCICore/Exceptions/DimensionMismatch.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Util/DebugStream.h>
#include <sstream>

#define MIKE_DEBUG

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

template <class T> class FlatAttrib : public DiscreteAttrib<T> 
{
public:
  
  //////////
  // Constructors
  FlatAttrib();
  FlatAttrib(int);
  FlatAttrib(int, int);
  FlatAttrib(int, int, int);
  FlatAttrib(const FlatAttrib& copy);
  
  //////////
  // Destructor
  virtual ~FlatAttrib();
  
  //////////
  // return the value at the given position
  const T &get1(int) const;
  const T &get2(int, int) const;
  const T &get3(int, int, int) const;

  T &set1(int x, T &val);
  T &set2(int x, int y, T &val);
  T &set3(int x, int y, int z, T &val);

  // TODO: begin, end 

  //////////
  // Resize the attribute to the specified dimensions
  virtual void resize(int);
  virtual void resize(int, int);
  virtual void resize(int, int, int);

  int size() const;


  virtual string get_info();  


  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

protected:
  vector<T> data;

  static DebugStream dbg;
};

}
}

namespace SCICore {
namespace Datatypes {

template <class T> DebugStream FlatAttrib<T>::dbg("FlatAttrib", true);

template <class T> PersistentTypeID FlatAttrib<T>::type_id("FlatAttrib", "Datatype", 0);


template <class T>
FlatAttrib<T>::FlatAttrib() :
  DiscreteAttrib<T>()
{
}

template <class T>
FlatAttrib<T>::FlatAttrib(int ix) :
  DiscreteAttrib<T>(ix), data(ix)
{
}

template <class T>
FlatAttrib<T>::FlatAttrib(int ix, int iy) :
  DiscreteAttrib<T>(ix, iy), data(ix * iy)
{
}

template <class T>
FlatAttrib<T>::FlatAttrib(int ix, int iy, int iz) :
  DiscreteAttrib<T>(ix, iy, iz), data(ix * iy * iz)
{
}

template <class T>
FlatAttrib<T>::FlatAttrib(const FlatAttrib& copy) :
  DiscreteAttrib<T>(copy), data(copy.data)
{
}


template <class T>
FlatAttrib<T>::~FlatAttrib()
{
}



template <class T> const T &
FlatAttrib<T>::get1(int ix) const
{
#ifdef MIKE_DEBUG
  if (dim != 1) {
    throw DimensionMismatch(1, dim);
  }
  if (ix >= nx) {
    throw ArrayIndexOutOfBounds(ix, 0, nx);
  }
#endif
  return data[ix];  
}

template <class T> const T &
FlatAttrib<T>::get2(int ix, int iy) const
{
#ifdef MIKE_DEBUG
  if (dim != 2) {
    throw DimensionMismatch(2, dim);
  }
  if (ix >= nx) {
    throw ArrayIndexOutOfBounds(ix, 0, nx);
  }
  if (iy >= ny) {
    throw ArrayIndexOutOfBounds(iy, 0, ny);
  }
#endif
  return data[iy*(nx)+ix];  
}

template <class T> const T &
FlatAttrib<T>::get3(int ix, int iy, int iz) const
{
#ifdef MIKE_DEBUG
  if (dim != 3) {
    throw DimensionMismatch(3, dim);
  }
  if(ix >= nx) {
    throw ArrayIndexOutOfBounds(ix, 0, nx);
  }
  if (iy >= ny) {
    throw ArrayIndexOutOfBounds(iy, 0, ny);
  }
  if (iz >= nz) {
    throw ArrayIndexOutOfBounds(iz, 0, nz);
  }
#endif
  return data[iz*(nx*ny)+iy*(nx)+ix];  
}



template <class T> T&
FlatAttrib<T>::set1(int ix, T& val)
{
#ifdef MIKE_DEBUG
  if (dim != 1) {
    throw DimensionMismatch(1, dim);
  }
  if (ix >= nx) {
    throw ArrayIndexOutOfBounds(ix, 0, nx);
  }
#endif
  return data[ix] = val;
}


template <class T> T&
FlatAttrib<T>::set2(int ix, int iy, T& val)
{
#ifdef MIKE_DEBUG
  if (dim != 2) {
    throw DimensionMismatch(2, dim);
  }
  if (ix >= nx) {
    throw ArrayIndexOutOfBounds(ix, 0, nx);
  }
  if (iy >= ny) {
    throw ArrayIndexOutOfBounds(iy, 0, ny);
  }
#endif
  return data[iy*(nx)+ix] = val;
}


template <class T> T&
FlatAttrib<T>::set3(int ix, int iy, int iz, T& val)
{
#ifdef MIKE_DEBUG
  if (dim != 3) {
    throw DimensionMismatch(3, dim);
  }
  if(ix >= nx) {
    throw ArrayIndexOutOfBounds(ix, 0, nx);
  }
  if (iy >= ny) {
    throw ArrayIndexOutOfBounds(iy, 0, ny);
  }
  if (iz >= nz) {
    throw ArrayIndexOutOfBounds(iz, 0, nz);
  }
#endif
  return data[iz*(nx*ny)+iy*(nx)+ix] = val;
}


// template <class T> bool FlatAttrib<T>::compute_minmax(){
//   has_minmax = 1;
//   if(data.empty()) {
//     min = 0;
//     max = 0;
//     return false;
//   }
//   else {
//     vector<T>::iterator itr;
//     T lmin = data[0];
//     T lmax = lmin;
//     for(itr = data.begin(); itr != data.end(); itr++){
//       lmin = Min(lmin, *itr);
//       lmax = Max(lmax, *itr);
//     }
//     min = (double) lmin;
//     max = (double) lmax;
//     return true;
//   }
// }

template <class T> void
FlatAttrib<T>::resize(int x, int y, int z)
{
  DiscreteAttrib<T>::resize(x, y, z);
  data.resize(x*y*z);
}


template <class T> void
FlatAttrib<T>::resize(int x, int y)
{
  DiscreteAttrib<T>::resize(x, y);
  data.resize(x*y);
}


template <class T> void
FlatAttrib<T>::resize(int x)
{
  DiscreteAttrib<T>::resize(x);
  data.resize(x);
}


template <class T> int
FlatAttrib<T>::size() const
{
  return data.size();
}


template <class T> string FlatAttrib<T>::get_info(){
  ostringstream retval;
  retval <<
    "Name = " << name << '\n' <<
    "Type = FlatAttrib" << '\n' <<
    "Size = " << size() << '\n' <<
    "Data = ";
  vector<T>::iterator itr = data.begin();
  for(;itr!=data.end(); itr++){
    retval << *itr << " ";
  }
  retval << endl;
  return retval.str();
}

template <class T> void FlatAttrib<T>::io(Piostream&){
}

}  // end Datatypes
}  // end SCICore



#endif  // SCI_project_FlatAttrib_h



