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

#define BITBOUND 0

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

template <class T> class BrickAttrib : public FlatAttrib<T> 
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

  virtual int iterate(AttribFunctor<T> &func);

  int size() const;

  virtual string getInfo();  


  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

protected:
#if BITBOUND
  static const int XBRICKSIZE = 4;
  static const int YBRICKSIZE = 2;
  static const int ZBRICKSIZE = 2;
#else
  static const int XBRICKBITS = 4;
  static const int YBRICKBITS = 2;
  static const int ZBRICKBITS = 2;

  static const int XBRICKSPACE = ((1 << XBRICKBITS) - 1);
  static const int YBRICKSPACE = ((1 << YBRICKBITS) - 1);
  static const int ZBRICKSPACE = ((1 << ZBRICKBITS) - 1);
#endif

  int xbrickcount, ybrickcount, zbrickcount;
  void update_brick_counts();

  unsigned int linearize(int x, int y);
  unsigned int linearize(int x, int y, int z);
};



template <class T> PersistentTypeID BrickAttrib<T>::type_id("BrickAttrib", "Datatype", 0);


template <class T> void
BrickAttrib<T>::update_brick_counts()
{
#if BITBOUND
  xbrickcount = d_nx / XBRICKSIZE;
  if (d_nx % XBRICKSIZE) xbrickcount++;

  ybrickcount = d_ny / YBRICKSIZE;
  if (d_ny % YBRICKSIZE) ybrickcount++;

  zbrickcount = d_nz / ZBRICKSIZE;
  if (d_nz % ZBRICKSIZE) zbrickcount++;
#else
  xbrickcount = d_nx >> XBRICKBITS;
  if (d_nx & XBRICKSPACE) xbrickcount++;

  ybrickcount = d_ny >> YBRICKBITS;
  if (d_ny & YBRICKSPACE) ybrickcount++;

  zbrickcount = d_nz >> ZBRICKBITS;
  if (d_nz & ZBRICKSPACE) zbrickcount++;
#endif  
}


template <class T>
BrickAttrib<T>::BrickAttrib() :
  FlatAttrib<T>()
{
  update_brick_counts();
}

template <class T>
BrickAttrib<T>::BrickAttrib(int x) :
  FlatAttrib<T>()
{
  resize(x);
}

template <class T>
BrickAttrib<T>::BrickAttrib(int x, int y) :
  FlatAttrib<T>()
{
  resize(x, y);
}

template <class T>
BrickAttrib<T>::BrickAttrib(int x, int y, int z) :
  FlatAttrib<T>(x, y, z)
{
  resize(x, y, z);
}

template <class T>
BrickAttrib<T>::BrickAttrib(const BrickAttrib& copy) :
  FlatAttrib<T>(copy)
{
  update_brick_counts();
}


template <class T>
BrickAttrib<T>::~BrickAttrib()
{
}

#if BITBOUND
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
#else
template <class T> unsigned int
BrickAttrib<T>::linearize(int x, int y)
{
  int xb = x >> XBRICKBITS;
  int yb = y >> YBRICKBITS;

  int xr = x & XBRICKSPACE;
  int yr = y & YBRICKSPACE;

  int brick = yb * xbrickcount + xb;
  int baddr = (yr << XBRICKBITS) + xr;

  return (brick << (XBRICKBITS + YBRICKBITS)) + baddr;
}

template <class T> unsigned int
BrickAttrib<T>::linearize(int x, int y, int z)
{
  int xb = x >> XBRICKBITS;
  int yb = y >> YBRICKBITS;
  int zb = z >> ZBRICKBITS;

  int xr = x & XBRICKSPACE;	
  int yr = y & YBRICKSPACE;
  int zr = z & ZBRICKSPACE;

  int brick = zb * xbrickcount * ybrickcount + yb * xbrickcount + xb;
  int baddr = (zr << (XBRICKBITS + YBRICKBITS)) + (yr << XBRICKBITS) + xr;

  return (brick << (XBRICKBITS + YBRICKBITS + ZBRICKBITS)) + baddr;
}
#endif


template <class T> T &
BrickAttrib<T>::fget1(int ix)
{
  ASSERTEQ(d_dim, 1);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  return d_data[ix];
}

template <class T> T &
BrickAttrib<T>::fget2(int ix, int iy)
{
  ASSERTEQ(d_dim, 2);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  return d_data[linearize(ix, iy)];
}

template <class T> T &
BrickAttrib<T>::fget3(int ix, int iy, int iz)
{
  ASSERTEQ(d_dim, 3);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  CHECKARRAYBOUNDS(iz, 0, d_nz);
  return d_data[linearize(ix, iy, iz)];
}


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
  ASSERTEQ(d_dim, 1);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  d_data[ix] = val;
}

template <class T> void
BrickAttrib<T>::fset2(int ix, int iy, const T& val)
{
  ASSERTEQ(d_dim, 2);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  d_data[linearize(ix, iy)] = val;
}


template <class T> void
BrickAttrib<T>::fset3(int ix, int iy, int iz, const T& val)
{
  ASSERTEQ(d_dim, 3);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  CHECKARRAYBOUNDS(iz, 0, d_nz);
  d_data[linearize(ix, iy, iz)] = val;
}


// Generic setters for Flat type
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
BrickAttrib<T>::resize(int x, int y, int z)
{
  DiscreteAttrib<T>::resize(x, y, z);
  update_brick_counts();
#if BITBOUND
  d_data.resize(xbrickcount * XBRICKSIZE *
	      ybrickcount * YBRICKSIZE *
	      zbrickcount * XBRICKSIZE);
#else
  d_data.resize((xbrickcount * ybrickcount * zbrickcount) <<
	      (XBRICKBITS + YBRICKBITS + ZBRICKBITS));
#endif
}


template <class T> void
BrickAttrib<T>::resize(int x, int y)
{
  DiscreteAttrib<T>::resize(x, y);
  update_brick_counts();
#if BITBOUND
  d_data.resize(xbrickcount * XBRICKSIZE *
	      ybrickcount * YBRICKSIZE);
#else
  d_data.resize((xbrickcount * ybrickcount) <<
	      (XBRICKBITS + YBRICKBITS));
#endif
}


template <class T> void
BrickAttrib<T>::resize(int x)
{
  DiscreteAttrib<T>::resize(x);
#if BITBOUND
  d_data.resize(xbrickcount * XBRICKSIZE);
#else
  d_data.resize(xbrickcount << XBRICKBITS);
#endif
}


template <class T> int
BrickAttrib<T>::size() const
{
  switch (d_dim)
    {
    case 3:
      return d_nx * d_ny * d_nz;
    case 2:
      return d_nx * d_ny;
    case 1:
      return d_nx;
    default:
      return 0;
    }
}


template <class T> int
BrickAttrib<T>::iterate(AttribFunctor<T> &func)
{
  if (d_dim == 3)
    {
      for (int i = 0; i < d_nz; i++)
	{
	  for (int j = 0; j < d_ny; j++)
	    {
	      for (int k = 0; k < d_nx; k++)
		{
		  func(fget3(k, j, i));
		}
	    }
	}
      return size();
    }
  else if (d_dim == 2)
    {
      for (int i = 0; i < d_ny; i++)
	{
	  for (int j = 0; j < d_nx; j++)
	    {
	      func(fget2(j, i));
	    }
	}
      return size();
    }
  else
    {
      return FlatAttrib<T>::iterate(func);
    }
}



template <class T> string
BrickAttrib<T>::getInfo()
{
  ostringstream retval;
  retval <<
    "Name = " << d_name << endl <<
    "Type = BrickAttrib" << endl <<
    "Dim = " << d_dim << ": " << d_nx << ' ' << d_ny << ' ' << d_nz << endl <<
    "Brickcounts = " 
	 << xbrickcount << ' ' << ybrickcount << ' ' << zbrickcount << endl <<
#if BITBOUND
#else
    "Bricksizes = "
	 << (1 << XBRICKBITS) << ' '
	 << (1 << YBRICKBITS) << ' '
	 << (1 << ZBRICKBITS) << endl <<
#endif
    "Size = " << size() << endl <<
    "Data = ";
  vector<T>::iterator itr = d_data.begin();
  int i = 0;
  for(;itr!=d_data.end() && i < 200; itr++, i++) {
    retval << *itr << " ";
  }
  if (itr != d_data.end()) { retval << "..."; }
  retval << endl;
  return retval.str();
}

template <class T> void BrickAttrib<T>::io(Piostream&){
}

}  // end Datatypes
}  // end SCICore



#endif  // SCI_project_BrickAttrib_h



