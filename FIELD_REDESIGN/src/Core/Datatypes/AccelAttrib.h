//  AccelAttrib.h - scalar attribute stored as a flat array
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
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
  AccelAttrib(int, int);
  AccelAttrib(int, int, int);
  AccelAttrib(const AccelAttrib<T> &copy);

  //////////
  // Destructor
  virtual ~AccelAttrib();
  
  virtual void get2(T &result, int x, int y);
  virtual void get3(T &result, int x, int y, int z);

  virtual T &get2(int x, int y);
  virtual T &get3(int x, int y, int z);

  //////////
  // return the value at the given position
  T& fget2(int, int);
  T& fget3(int, int, int);


  virtual void set2(int x, int y, const T &val);
  virtual void set3(int x, int y, int z, const T &val);
  void fset2(int x, int y, const T &val);
  void fset3(int x, int y, int z, const T &val);


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

  
private:
  void create_accel_structure();

  // Accel structure.
  vector<vector<T *> > accel3;
  vector<T *> accel2;
};



template <class T>
void
AccelAttrib<T>::create_accel_structure()
{
  if (dim == 3)
    {
      accel3.resize(nz);
      for (int i=0; i < nz; i++)
	{
	  accel3[i].resize(ny);
	  for (int j=0; j < ny; j++)
	    {
	      accel3[i][j] = &data[i*nx*ny + j*nx];
	    }
	}
    }
  else if (dim == 2)
    {
      accel2.resize(ny);
      for (int i=0; i < ny; i++)
	{
	  accel2[i] = &data[i*nx];
	}
    }
}


template <class T>
AccelAttrib<T>::AccelAttrib()
  : FlatAttrib<T>()
{
}

template <class T>
AccelAttrib<T>::AccelAttrib(int ix, int iy)
  : FlatAttrib<T>(ix, iy)
{
  create_accel_structure();
}


template <class T>
AccelAttrib<T>::AccelAttrib(int ix, int iy, int iz)
  : FlatAttrib<T>(ix, iy, iz)
{
  create_accel_structure();
}


template <class T>
AccelAttrib<T>::AccelAttrib(const AccelAttrib<T>& copy)
  : FlatAttrib<T>(copy)
{
  create_accel_structure();
}

template <class T> T &
AccelAttrib<T>::fget2(int ix, int iy)
{
  // assert(dim == 2, DimensionMismatch(dim, 2));
  return accel2[iy][ix];
}

template <class T> T &
AccelAttrib<T>::fget3(int ix, int iy, int iz)
{
  // assert(dim == 3, DimensionMismatch(dim, 3));
  return accel3[iz][iy][ix];
}


// Copy wrappers, no allocation of result.
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
AccelAttrib<T>::fset2(int x, int y, const T &val)
{
  // assert(dim == 2, DimensionMismatch(dim, 2));
  accel2[y][x] = val;
}

template <class T> void
AccelAttrib<T>::fset3(int x, int y, int z, const T &val)
{
  // assert(dim == 3, DimensionMismatch(dim, 3));
  accel3[z][y][x] = val;
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



template <class T>
void
AccelAttrib<T>::resize(int)
{
  throw DimensionMismatch(1, 3);
}


template <class T>
void
AccelAttrib<T>::resize(int x, int y)
{
  FlatAttrib<T>::resize(x, y);
  create_accel_structure();
}


template <class T>
void
AccelAttrib<T>::resize(int x, int y, int z)
{
  FlatAttrib<T>::resize(x, y, z);
  create_accel_structure();
}



template <class T>
PersistentTypeID AccelAttrib<T>::type_id("AccelAttrib", "Datatype", 0);


template <class T>
string
AccelAttrib<T>::get_info()
{
  ostringstream retval;
  retval <<
    "Name = " << name << '\n' <<
    "Type = AccelAttrib" << '\n' <<
    "Dim = " << dim << ": " << nx << ' ' << ny << ' ' << nz << '\n' <<
    "Size = " << size() << '\n' <<
    "Data = ";
  vector<T>::iterator itr = data.begin();
  for ( ;itr!=data.end(); itr++)
    {
      retval << *itr << " ";
    }
  retval << endl;
  return retval.str();
}

template <class T>
void
AccelAttrib<T>::io(Piostream&)
{
}

}  // end Datatypes
}  // end SCICore



#endif  // SCI_project_AccelAttrib_h







