//  FlatAttrib.h - scalar attribute stored as a flat array
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_FlatAttrib_h
#define SCI_project_FlatAttrib_h 1

#include <vector>

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/DiscreteAttrib.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Exceptions/ArrayIndexOutOfBounds.h>
#include <Core/Exceptions/DimensionMismatch.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Util/DebugStream.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Persistent/PersistentSTL.h>

#ifdef __sgi
#include <limits>
#endif
#include <sstream>
#include <memory.h>

namespace SCIRun {

using std::vector;
using std::ostringstream;

template <class T> class FlatAttrib : public DiscreteAttrib<T> 
{
public:
  
  //////////
  // Constructors
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

  int size() const;

  virtual string getInfo();
  virtual string getTypeName(int=0);

  virtual int iterate(AttribFunctor<T> &func);

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static string typeName(int);
  static Persistent* maker();

protected:
  
  // GROUP: Private data
  //////////
  // 
  vector<T>       data_;
  static DebugStream dbg;
};

//////////
// PIO support
template <class T> Persistent*
FlatAttrib<T>::maker(){
  return new FlatAttrib<T>(0);
}

template <class T>
string FlatAttrib<T>::typeName(int n){
  ASSERTRANGE(n, 0, 2);
  static string t1name    = findTypeName((T*)0);
  static string className = string("FlatAttrib<") + t1name +">";
  
  switch (n){
  case 1:
    return t1name;
  default:
    return className;
  }
}

template <class T> 
PersistentTypeID FlatAttrib<T>::type_id(FlatAttrib<T>::typeName(0), 
					DiscreteAttrib<T>::typeName(0), 
					FlatAttrib<T>::maker);

#define FLATATTRIB_VERSION 1

template <class T> void
FlatAttrib<T>::io(Piostream& stream)
{
  stream.begin_class(typeName(0).c_str(), FLATATTRIB_VERSION);
  
  // -- base class PIO
  DiscreteAttrib<T>::io(stream);
  Pio(stream, data_);
  stream.end_class();
}


template <class T> DebugStream FlatAttrib<T>::dbg("FlatAttrib", true);

template <class T>
FlatAttrib<T>::FlatAttrib(int ix) :
  DiscreteAttrib<T>(ix), data_(ix)
{
}

template <class T>
FlatAttrib<T>::FlatAttrib(int ix, int iy) :
  DiscreteAttrib<T>(ix, iy), data_(ix * iy)
{
}

template <class T>
FlatAttrib<T>::FlatAttrib(int ix, int iy, int iz) :
  DiscreteAttrib<T>(ix, iy, iz), data_(ix * iy * iz)
{
}

template <class T>
FlatAttrib<T>::FlatAttrib(const FlatAttrib& copy) :
  DiscreteAttrib<T>(copy), data_(copy.data_)
{
}

template <class T>
FlatAttrib<T>::~FlatAttrib()
{
}

template <class T> T &
FlatAttrib<T>::fget1(int ix)
{
  ASSERTEQ(dim_, 1);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  return data_[ix];  
}

template <class T> T &
FlatAttrib<T>::fget2(int ix, int iy)
{
  ASSERTEQ(dim_, 2);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  return data_[iy*(nx_)+ix];  
}

template <class T> T &
FlatAttrib<T>::fget3(int ix, int iy, int iz)
{
  ASSERTEQ(dim_, 3);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  CHECKARRAYBOUNDS(iz, 0, nz_);
  return data_[iz*(nx_*ny_)+iy*(nx_)+ix];  
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
  ASSERTEQ(dim_, 1);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  data_[ix] = val;
}


template <class T> void
FlatAttrib<T>::fset2(int ix, int iy, const T& val)
{
  ASSERTEQ(dim_, 2);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  data_[iy*(nx_)+ix] = val;
}


template <class T> void
FlatAttrib<T>::fset3(int ix, int iy, int iz, const T& val)
{
  ASSERTEQ(dim_, 3);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  CHECKARRAYBOUNDS(iz, 0, nz_);
  data_[iz*(nx_*ny_)+iy*(nx_)+ix] = val;
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
//   if(data_.empty()) {
//     min = 0;
//     max = 0;
//     return false;
//   }
//   else {
//     vector<T>::iterator itr;
//     T lmin = data_[0];
//     T lmax = lmin;
//     for(itr = data_.begin(); itr != data_.end(); itr++){
//       lmin = Min(lmin, *itr);
//       lmax = Max(lmax, *itr);
//     }
//     min = (double) lmin;
//     max = (double) lmax;
//     return true;
//   }
// }

template <class T> int
FlatAttrib<T>::size() const
{
  return data_.size();
}


template <class T> int
FlatAttrib<T>::iterate(AttribFunctor<T> &func)
{
  vector<T>::iterator itr = data_.begin();
  while (itr != data_.end())
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
    "Name = " << name_ << endl <<
    "Type = FlatAttrib" << endl <<
    "Dim = " << dim_ << ": " << nx_ << ' ' << ny_ << ' ' << nz_ << endl <<
    "Size = " << size() << endl <<
    "Data = ";
  vector<T>::iterator itr = data_.begin();
  int i = 0;
  for(;itr!=data_.end() && i < 1000; itr++, i++)  {
    retval << *itr << " ";
  }
  if (itr != data_.end()) { retval << "..."; }
  retval << endl;
  return retval.str();
}

template <class T> string
FlatAttrib<T>::getTypeName(int n){
  return typeName(n);
}

} // End namespace SCIRun

#endif  // SCI_project_FlatAttrib_h
