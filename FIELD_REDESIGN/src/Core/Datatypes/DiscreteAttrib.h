//  DiscreteAttrib.h
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
//
//  Copyright (C) 2000 SCI Institute
//
//  Attribute containing a finite number of discrete values.
//

#ifndef SCI_project_DiscreteAttrib_h
#define SCI_project_DiscreteAttrib_h 1

#include <vector>

#include <SCICore/Datatypes/Attrib.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore{
namespace Datatypes{

using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Point;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

template <class T> class AttribFunctor
{
public:
  virtual void operator() (T &) {}
};

template <class T> class DiscreteAttrib : public Attrib //abstract class
{
public:
  DiscreteAttrib();
  DiscreteAttrib(int x);
  DiscreteAttrib(int x, int y);
  DiscreteAttrib(int x, int y, int z);
  
  DiscreteAttrib(const DiscreteAttrib& copy);

  virtual ~DiscreteAttrib();

  virtual void get1(T &result, int x);
  virtual void get2(T &result, int x, int y);
  virtual void get3(T &result, int x, int y, int z);

  virtual T &get1(int x);
  virtual T &get2(int x, int y);
  virtual T &get3(int x, int y, int z);

  T &fget1(int x);
  T &fget2(int x, int y);
  T &fget3(int x, int y, int z);
  

  virtual void set1(int x, const T &val);
  virtual void set2(int x, int y, const T &val);
  virtual void set3(int x, int y, int z, const T &val);

  void fset1(int x, const T &val);
  void fset2(int x, int y, const T &val);
  void fset3(int x, int y, int z, const T &val);

  // Implement begin()
  // Implement end()

  // Resize the attribute to the specified dimensions
  virtual void resize(int, int, int);
  virtual void resize(int, int);
  virtual void resize(int);

  int size() const;


  virtual string get_info();  

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  virtual int iterate(AttribFunctor<T> &func);

  virtual int xsize() const { return nx; }
  virtual int ysize() const { return ny; }
  virtual int zsize() const { return nz; }
  virtual int dimension() const { return dim; }

protected:
  /////////
  // an identifying name
  int nx, ny, nz;
  int dim;

private:
  T defval;
};


template <class T>
DiscreteAttrib<T>::DiscreteAttrib() :
  Attrib(), nx(0), ny(0), nz(0), dim(0)
{
}

template <class T>
DiscreteAttrib<T>::DiscreteAttrib(int ix) :
  Attrib(), nx(ix), ny(0), nz(0), dim(1)
{
}

template <class T>
DiscreteAttrib<T>::DiscreteAttrib(int ix, int iy) :
  Attrib(), nx(ix), ny(iy), nz(0), dim(2)
{
}

template <class T>
DiscreteAttrib<T>::DiscreteAttrib(int ix, int iy, int iz) :
  Attrib(), nx(ix), ny(iy), nz(iz), dim(3)
{
}

template <class T>
DiscreteAttrib<T>::DiscreteAttrib(const DiscreteAttrib& copy) :
  Attrib(copy), nx(copy.nx), ny(copy.ny), nz(copy.nz), dim(copy.dim)
{
}


template <class T>
DiscreteAttrib<T>::~DiscreteAttrib()
{
}


template <class T> T &
DiscreteAttrib<T>::fget1(int)
{
  return defval;
}


template <class T> T &
DiscreteAttrib<T>::fget2(int, int)
{
  return defval;
}


template <class T> T &
DiscreteAttrib<T>::fget3(int, int, int)
{
  return defval;
}


// Copy wrappers, no allocation of result.
template <class T> void
DiscreteAttrib<T>::get1(T &result, int ix)
{
  result = fget1(ix);
}

template <class T> void
DiscreteAttrib<T>::get2(T &result, int ix, int iy)
{
  result = fget2(ix, iy);
}

template <class T> void
DiscreteAttrib<T>::get3(T &result, int ix, int iy, int iz)
{
  result = fget3(ix, iy, iz);
}


// Virtual wrappers for inline functions.
template <class T> T &
DiscreteAttrib<T>::get1(int ix)
{
  return fget1(ix);
}

template <class T> T &
DiscreteAttrib<T>::get2(int ix, int iy)
{
  return fget2(ix, iy);
}

template <class T> T &
DiscreteAttrib<T>::get3(int ix, int iy, int iz)
{
  return fget3(ix, iy, iz);
}




template <class T> void
DiscreteAttrib<T>::fset1(int, const T &val)
{
  defval = val;
}


template <class T> void
DiscreteAttrib<T>::fset2(int, int, const T &val)
{
  defval = val;
}


template <class T> void
DiscreteAttrib<T>::fset3(int, int, int, const T &val)
{
  defval = val;
}


// Generic setters for Discrete type
template <class T> void
DiscreteAttrib<T>::set1(int x, const T &val)
{
  fset1(x, val);
}

template <class T> void
DiscreteAttrib<T>::set2(int x, int y, const T &val)
{
  fset2(x, y, val);
}

template <class T> void
DiscreteAttrib<T>::set3(int x, int y, int z, const T &val)
{
  fset3(x, y, z, val);
}


// template <class T> bool DiscreteAttrib<T>::compute_minmax(){
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
DiscreteAttrib<T>::resize(int x, int y, int z)
{
  nx = x;
  ny = y;
  nz = z;
  dim = 3;
}


template <class T> void
DiscreteAttrib<T>::resize(int x, int y)
{
  nx = x;
  ny = y;
  nz = 0;
  dim = 2;
}


template <class T> void
DiscreteAttrib<T>::resize(int x)
{
  nx = x;
  ny = 0;
  nz = 0;
  dim = 1;
}


template <class T> int
DiscreteAttrib<T>::iterate(AttribFunctor<T> &func)
{
  func(defval);
  return 1;
}


template <class T> int
DiscreteAttrib<T>::size() const
{
  switch (dim)
    {
    case 1:
      return nx;

    case 2:
      return nx * ny;

    case 3:
      return nx * ny * nz;

    default:
      return 0;
    }
}


template <class T> PersistentTypeID DiscreteAttrib<T>::type_id("DiscreteAttrib", "Datatype", 0);


template <class T> string
DiscreteAttrib<T>::get_info()
{
  ostringstream retval;
  retval <<
    "Name = " << name << '\n' <<
    "Type = DiscreteAttrib" << '\n' <<
    "Dim = " << dim << ": " << nx << ' ' << ny << ' ' << nz << '\n' <<
    "Size = " << size() << '\n';
  return retval.str();
}

template <class T> void
DiscreteAttrib<T>::io(Piostream&)
{
}


}  // end Datatypes
}  // end SCICore

#endif



