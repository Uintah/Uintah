//  AccelAttrib.h - scalar attribute stored as a Accel array
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_AccelAttrib_h
#define SCI_project_AccelAttrib_h 1

#include <SCICore/Datatypes/FlatAttrib.h>


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

template <class T> class AccelAttrib : public FlatAttrib<T> 
{
public:
  
  //////////
  // Constructors
  AccelAttrib();
  AccelAttrib(int);
  AccelAttrib(int, int);
  AccelAttrib(int, int, int);
  AccelAttrib(const AccelAttrib& copy);
  
  //////////
  // Destructor
  virtual ~AccelAttrib();
  

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

  virtual string get_info();  

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

protected:
  void create_accel_structure();

  // Accel structure.
  vector<vector<T *> > accel3;
  vector<T *> accel2;
};



template <class T> PersistentTypeID AccelAttrib<T>::type_id("AccelAttrib", "Datatype", 0);


template <class T>
void
AccelAttrib<T>::create_accel_structure()
{
  data.reserve(data.size());
  if (dim == 3)
    {
      accel3.resize(nz);
      for (int i=0; i < nz; i++)
	{
	  accel3[i].resize(ny);
	  for (int j=0; j < ny; j++)
	    {
	      accel3[i][j] = &(data[i*nx*ny + j*nx]);
	    }
	}
    }
  else if (dim == 2)
    {
      accel2.resize(ny);
      for (int i=0; i < ny; i++)
	{
	  accel2[i] = &(data[i*nx]);
	}
    }
}



template <class T>
AccelAttrib<T>::AccelAttrib() :
  FlatAttrib<T>()
{
  create_accel_structure();
}

template <class T>
AccelAttrib<T>::AccelAttrib(int x) :
  FlatAttrib<T>(x)
{
  create_accel_structure();
}

template <class T>
AccelAttrib<T>::AccelAttrib(int x, int y) :
  FlatAttrib<T>(x, y)
{
  create_accel_structure();
}

template <class T>
AccelAttrib<T>::AccelAttrib(int x, int y, int z) :
  FlatAttrib<T>(x, y, z)
{
  create_accel_structure();
}

template <class T>
AccelAttrib<T>::AccelAttrib(const AccelAttrib& copy) :
  FlatAttrib<T>(copy)
{
  create_accel_structure();
}


template <class T>
AccelAttrib<T>::~AccelAttrib()
{
}


template <class T> T &
AccelAttrib<T>::fget1(int ix)
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
AccelAttrib<T>::fget2(int ix, int iy)
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
  return accel2[iy][ix];  
}

template <class T> T &
AccelAttrib<T>::fget3(int ix, int iy, int iz)
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
  return accel3[iz][iy][ix];  
}


// Copy wrappers, no allocation of result.
template <class T> void
AccelAttrib<T>::get1(T &result, int ix)
{
  result = fget1(ix);
}

template <class T> void
AccelAttrib<T>::get2(T &result, int ix, int iy)
{
  result = fget2(ix, iy);
}

template <class T> void
AccelAttrib<T>::get3(T &result, int ix, int iy, int iz)
{
  result = fget3(ix, iy, iz);
}


// Virtual wrappers for inline functions.
template <class T> T &
AccelAttrib<T>::get1(int ix)
{
  return fget1(ix);
}

template <class T> T &
AccelAttrib<T>::get2(int ix, int iy)
{
  return fget2(ix, iy);
}

template <class T> T &
AccelAttrib<T>::get3(int ix, int iy, int iz)
{
  return fget3(ix, iy, iz);
}



template <class T> void
AccelAttrib<T>::fset1(int ix, const T& val)
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
AccelAttrib<T>::fset2(int ix, int iy, const T& val)
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
  accel2[iy][ix] = val;
}


template <class T> void
AccelAttrib<T>::fset3(int ix, int iy, int iz, const T& val)
{
#ifdef MIKE_DEBUG
  if (dim != 3) {
    throw DimensionMismatch(3, dim);
  }
  if (ix >= nx) {
    throw ArrayIndexOutOfBounds(ix, 0, nx);
  }
  if (iy >= ny) {
    throw ArrayIndexOutOfBounds(iy, 0, ny);
  }
  if (iz >= nz) {
    throw ArrayIndexOutOfBounds(iz, 0, nz);
  }
#endif
  accel3[iz][iy][ix] = val;
}


// Generic setters for Flat type
template <class T> void
AccelAttrib<T>::set1(int x, const T &val)
{
  fset1(x, val);
}

template <class T> void
AccelAttrib<T>::set2(int x, int y, const T &val)
{
  fset2(x, y, val);
}

template <class T> void
AccelAttrib<T>::set3(int x, int y, int z, const T &val)
{
  fset3(x, y, z, val);
}

// template <class T> bool AccelAttrib<T>::compute_minmax(){
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
AccelAttrib<T>::resize(int x, int y, int z)
{
  FlatAttrib<T>::resize(x, y, z);
  create_accel_structure();
}


template <class T> void
AccelAttrib<T>::resize(int x, int y)
{
  FlatAttrib<T>::resize(x, y);
  create_accel_structure();
}


template <class T> void
AccelAttrib<T>::resize(int x)
{
  FlatAttrib<T>::resize(x);
  create_accel_structure();
}


template <class T> string AccelAttrib<T>::get_info() {
  ostringstream retval;
  retval <<
    "Name = " << name << endl <<
    "Type = AccelAttrib" << endl <<
    "Dim = " << dim << ": " << nx << ' ' << ny << ' ' << nz << endl <<
    "Size = " << size() << endl;
#if 1
  retval << "Data = ";
  vector<T>::iterator itr = data.begin();
  int i = 0;
  for(;itr!=data.end() && i < 1000; itr++, i++) {
    retval << *itr << " ";
  }
  if (itr != data.end()) { retval << "..."; }
  retval << endl;
#else
  for (int k = 0; k < nz; k++)
    {
      for (int j = 0; j < nz; j++)
	{
	  retval << "  " << &(data[k * nx*ny + j * nx]);
	}
      retval << endl;
    }
  retval << endl;
#endif
  return retval.str();
}

template <class T> void AccelAttrib<T>::io(Piostream&){
}

}  // end Datatypes
}  // end SCICore



#endif  // SCI_project_AccelAttrib_h



