//  BrickAttrib.h - scalar attribute stored as a bricked array
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_BrickAttrib_h
#define SCI_project_BrickAttrib_h 1

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

template <class T> class BrickAttrib : public DiscreteAttrib<T> 
{
public:
  
  //////////
  // Constructors
  BrickAttrib();
  BrickAttrib(int);
  BrickAttrib(int, int);
  BrickAttrib(int, int, int);
  BrickAttrib(const BrickAttrib& copy);
  
  //////////
  // Destructor
  virtual ~BrickAttrib();
  

  virtual void get1(T &result, int x);
  virtual void get2(T &result, int x, int y);
  virtual void get3(T &result, int x, int y, int z);

  virtual T &get1(int x);
  virtual T &get2(int x, int y);
  virtual T &get3(int x, int y, int z);


  //////////
  // return the value at the given position
  T &fget1(int);
  T &fget2(int, int);
  T &fget3(int, int, int);

  virtual void set1(int x, const T &val);
  virtual void set2(int x, int y, const T &val);
  virtual void set3(int x, int y, int z, const T &val);

  void fset1(int x, const T &val);
  void fset2(int x, int y, const T &val);
  void fset3(int x, int y, int z, const T &val);

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

  static const int XBRICKSIZE = 256;
  static const int YBRICKSIZE = 512;
  static const int ZBRICKSIZE = 1024;
  int xbrickcount, ybrickcount, zbrickcount;

  void update_brick_counts();

  unsigned int linearize(int x, int y);
  unsigned int linearize(int x, int y, int z);

  static DebugStream dbg;
};



template <class T> DebugStream BrickAttrib<T>::dbg("BrickAttrib", true);

template <class T> PersistentTypeID BrickAttrib<T>::type_id("BrickAttrib", "Datatype", 0);


template <class T> void
BrickAttrib<T>::update_brick_counts()
{
  xbrickcount = nx / XBRICKSIZE;
  if (nx % XBRICKSIZE) xbrickcount++;

  ybrickcount = ny / YBRICKSIZE;
  if (ny % YBRICKSIZE) ybrickcount++;

  zbrickcount = nz / ZBRICKSIZE;
  if (nz % ZBRICKSIZE) zbrickcount++;
}


template <class T>
BrickAttrib<T>::BrickAttrib() :
  DiscreteAttrib<T>()
{
  update_brick_counts();
}

template <class T>
BrickAttrib<T>::BrickAttrib(int ix) :
  DiscreteAttrib<T>(ix), data(ix)
{
  update_brick_counts();
}

template <class T>
BrickAttrib<T>::BrickAttrib(int ix, int iy) :
  DiscreteAttrib<T>(ix, iy), data(ix * iy)
{
  update_brick_counts();
}

template <class T>
BrickAttrib<T>::BrickAttrib(int ix, int iy, int iz) :
  DiscreteAttrib<T>(ix, iy, iz), data(ix * iy * iz)
{
  update_brick_counts();
}

template <class T>
BrickAttrib<T>::BrickAttrib(const BrickAttrib& copy) :
  DiscreteAttrib<T>(copy), data(copy.data)
{
  update_brick_counts();
}


template <class T>
BrickAttrib<T>::~BrickAttrib()
{
}


template <class T> unsigned int
BrickAttrib<T>::linearize(int x, int y)
{
  int xb = x / XBRICKSIZE;
  int yb = y / YBRICKSIZE;

  int xr = x % XBRICKSIZE;	
  int yr = y % YBRICKSIZE;

  int brick = yb * xbrickcount + xb;
  int baddr = yr * XBRICKSIZE + xr;

  return brick * (XBRICKSIZE * YBRICKSIZE) + baddr;
}

template <class T> unsigned int
BrickAttrib<T>::linearize(int x, int y, int z)
{
  int xb = x / XBRICKSIZE;
  int yb = y / YBRICKSIZE;
  int zb = z / ZBRICKSIZE;

  int xr = x % XBRICKSIZE;	
  int yr = y % YBRICKSIZE;
  int zr = z % ZBRICKSIZE;

  int brick = zb * xbrickcount * ybrickcount + yb * xbrickcount + xb;
  int baddr = zr * (XBRICKSIZE * YBRICKSIZE) + yr * XBRICKSIZE + xr;

  return brick * (XBRICKSIZE * YBRICKSIZE * ZBRICKSIZE) + baddr;
}



template <class T> T &
BrickAttrib<T>::fget1(int ix)
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

template <class T> T &
BrickAttrib<T>::fget2(int ix, int iy)
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
  return data[linearize(ix, iy)];
}

template <class T> T &
BrickAttrib<T>::fget3(int ix, int iy, int iz)
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
  return data[linearize(ix, iy, iz)];
}


// Copy wrappers, no allocation of result.
template <class T> void
BrickAttrib<T>::get1(T &result, int ix)
{
  result = fget1(ix);
}

template <class T> void
BrickAttrib<T>::get2(T &result, int ix, int iy)
{
  result = fget2(ix, iy);
}

template <class T> void
BrickAttrib<T>::get3(T &result, int ix, int iy, int iz)
{
  result = fget3(ix, iy, iz);
}


// Virtual wrappers for inline functions.
template <class T> T &
BrickAttrib<T>::get1(int ix)
{
  return fget1(ix);
}

template <class T> T &
BrickAttrib<T>::get2(int ix, int iy)
{
  return fget2(ix, iy);
}

template <class T> T &
BrickAttrib<T>::get3(int ix, int iy, int iz)
{
  return fget3(ix, iy, iz);
}



template <class T> void
BrickAttrib<T>::fset1(int ix, const T& val)
{
#ifdef MIKE_DEBUG
  if (dim != 1) {
    throw DimensionMismatch(1, dim);
  }
  if (ix >= nx) {
    throw ArrayIndexOutOfBounds(ix, 0, nx);
  }
#endif
  data[ix] = val;
}


template <class T> void
BrickAttrib<T>::fset2(int ix, int iy, const T& val)
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
  data[linearize(ix, iy)] = val;
}


template <class T> void
BrickAttrib<T>::fset3(int ix, int iy, int iz, const T& val)
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
  data[linearize(ix, iy, iz)] = val;
}


// Generic setters for Discrete type
template <class T> void
BrickAttrib<T>::set1(int x, const T &val)
{
  fset1(x, val);
}

template <class T> void
BrickAttrib<T>::set2(int x, int y, const T &val)
{
  fset2(x, y, val);
}

template <class T> void
BrickAttrib<T>::set3(int x, int y, int z, const T &val)
{
  fset3(x, y, z, val);
}

// template <class T> bool BrickAttrib<T>::compute_minmax(){
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
BrickAttrib<T>::resize(int x, int y, int z)
{
  DiscreteAttrib<T>::resize(x, y, z);
  data.resize(x*y*z);
  update_brick_counts();
}


template <class T> void
BrickAttrib<T>::resize(int x, int y)
{
  DiscreteAttrib<T>::resize(x, y);
  data.resize(x*y);
  update_brick_counts();
}


template <class T> void
BrickAttrib<T>::resize(int x)
{
  DiscreteAttrib<T>::resize(x);
  data.resize(x);
  update_brick_counts();
}


template <class T> int
BrickAttrib<T>::size() const
{
  return data.size();
}


template <class T> string BrickAttrib<T>::get_info(){
  ostringstream retval;
  retval <<
    "Name = " << name << '\n' <<
    "Type = BrickAttrib" << '\n' <<
    "Dim = " << dim << ": " << nx << ' ' << ny << ' ' << nz << '\n' <<
    "Size = " << size() << '\n' <<
    "Data = ";
  vector<T>::iterator itr = data.begin();
  for(;itr!=data.end(); itr++){
    retval << *itr << " ";
  }
  retval << endl;
  return retval.str();
}

template <class T> void BrickAttrib<T>::io(Piostream&){
}

}  // end Datatypes
}  // end SCICore



#endif  // SCI_project_BrickAttrib_h



