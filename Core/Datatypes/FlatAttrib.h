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


  virtual string getInfo();

  virtual int iterate(AttribFunctor<T> &func);

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

protected:
  vector<T> d_data;

  static DebugStream dbg;
};



template <class T> DebugStream FlatAttrib<T>::dbg("FlatAttrib", true);

template <class T> PersistentTypeID FlatAttrib<T>::type_id("FlatAttrib", "Datatype", 0);


template <class T>
FlatAttrib<T>::FlatAttrib() :
  DiscreteAttrib<T>()
{
}

template <class T>
FlatAttrib<T>::FlatAttrib(int ix) :
  DiscreteAttrib<T>(ix), d_data(ix)
{
}

template <class T>
FlatAttrib<T>::FlatAttrib(int ix, int iy) :
  DiscreteAttrib<T>(ix, iy), d_data(ix * iy)
{
}

template <class T>
FlatAttrib<T>::FlatAttrib(int ix, int iy, int iz) :
  DiscreteAttrib<T>(ix, iy, iz), d_data(ix * iy * iz)
{
}

template <class T>
FlatAttrib<T>::FlatAttrib(const FlatAttrib& copy) :
  DiscreteAttrib<T>(copy), d_data(copy.d_data)
{
}


template <class T>
FlatAttrib<T>::~FlatAttrib()
{
}


template <class T> T &
FlatAttrib<T>::fget1(int ix)
{
  ASSERTEQ(d_dim, 1);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  return d_data[ix];  
}

template <class T> T &
FlatAttrib<T>::fget2(int ix, int iy)
{
  ASSERTEQ(d_dim, 2);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  return d_data[iy*(d_nx)+ix];  
}

template <class T> T &
FlatAttrib<T>::fget3(int ix, int iy, int iz)
{
  ASSERTEQ(d_dim, 3);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  CHECKARRAYBOUNDS(iz, 0, d_nz);
  return d_data[iz*(d_nx*d_ny)+iy*(d_nx)+ix];  
}


// Copy wrappers, no allocation of result.
template <class T> void
FlatAttrib<T>::get1(T &result, int ix)
{
  result = fget1(ix);
}

template <class T> void
FlatAttrib<T>::get2(T &result, int ix, int iy)
{
  result = fget2(ix, iy);
}

template <class T> void
FlatAttrib<T>::get3(T &result, int ix, int iy, int iz)
{
  result = fget3(ix, iy, iz);
}


// Virtual wrappers for inline functions.
template <class T> T &
FlatAttrib<T>::get1(int ix)
{
  return fget1(ix);
}

template <class T> T &
FlatAttrib<T>::get2(int ix, int iy)
{
  return fget2(ix, iy);
}

template <class T> T &
FlatAttrib<T>::get3(int ix, int iy, int iz)
{
  return fget3(ix, iy, iz);
}



template <class T> void
FlatAttrib<T>::fset1(int ix, const T& val)
{
  ASSERTEQ(d_dim, 1);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  d_data[ix] = val;
}


template <class T> void
FlatAttrib<T>::fset2(int ix, int iy, const T& val)
{
  ASSERTEQ(d_dim, 2);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  d_data[iy*(d_nx)+ix] = val;
}


template <class T> void
FlatAttrib<T>::fset3(int ix, int iy, int iz, const T& val)
{
  ASSERTEQ(d_dim, 3);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  CHECKARRAYBOUNDS(iz, 0, d_nz);
  d_data[iz*(d_nx*d_ny)+iy*(d_nx)+ix] = val;
}


// Generic setters for Discrete type
template <class T> void
FlatAttrib<T>::set1(int x, const T &val)
{
  fset1(x, val);
}

template <class T> void
FlatAttrib<T>::set2(int x, int y, const T &val)
{
  fset2(x, y, val);
}

template <class T> void
FlatAttrib<T>::set3(int x, int y, int z, const T &val)
{
  fset3(x, y, z, val);
}

// template <class T> bool FlatAttrib<T>::compute_minmax(){
//   has_minmax = 1;
//   if(d_data.empty()) {
//     min = 0;
//     max = 0;
//     return false;
//   }
//   else {
//     vector<T>::iterator itr;
//     T lmin = d_data[0];
//     T lmax = lmin;
//     for(itr = d_data.begin(); itr != d_data.end(); itr++){
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
  d_data.resize(x*y*z);
}


template <class T> void
FlatAttrib<T>::resize(int x, int y)
{
  DiscreteAttrib<T>::resize(x, y);
  d_data.resize(x*y);
}


template <class T> void
FlatAttrib<T>::resize(int x)
{
  DiscreteAttrib<T>::resize(x);
  d_data.resize(x);
}


template <class T> int
FlatAttrib<T>::size() const
{
  return d_data.size();
}


template <class T> int
FlatAttrib<T>::iterate(AttribFunctor<T> &func)
{
  vector<T>::iterator itr = d_data.begin();
  while (itr != d_data.end())
    {
      func(*itr++);
    }
  return size();
}


template <class T> string
FlatAttrib<T>::getInfo()
{
  ostringstream retval;
  retval <<
    "Name = " << d_name << endl <<
    "Type = FlatAttrib" << endl <<
    "Dim = " << d_dim << ": " << d_nx << ' ' << d_ny << ' ' << d_nz << endl <<
    "Size = " << size() << endl <<
    "Data = ";
  vector<T>::iterator itr = d_data.begin();
  int i = 0;
  for(;itr!=d_data.end() && i < 1000; itr++, i++)  {
    retval << *itr << " ";
  }
  if (itr != d_data.end()) { retval << "..."; }
  retval << endl;
  return retval.str();
}

template <class T> void
FlatAttrib<T>::io(Piostream &stream)
{
  DiscreteAttrib<T>::io(stream);

  stream.begin_class("FlatAttrib", 0);
  for (int i=0; i < d_data.size(); i++)
    {
      //Pio(stream, d_data[i]);
    }
  stream.end_class();
}

}  // end Datatypes
}  // end SCICore



#endif  // SCI_project_FlatAttrib_h



