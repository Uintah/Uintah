//  BrickAttrib.h - scalar attribute stored as a bricked array
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_BrickAttrib_h
#define SCI_project_BrickAttrib_h 1

#include <vector>

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/DiscreteAttrib.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Exceptions/ArrayIndexOutOfBounds.h>
#include <Core/Exceptions/DimensionMismatch.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Util/DebugStream.h>
#include <sstream>

#define BITBOUND 0

namespace SCIRun {

using std::vector;
using std::ostringstream;

template <class T> class BrickAttrib : public FlatAttrib<T> 
{
public:
  
  //////////
  // Constructors
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

  virtual int iterate(AttribFunctor<T> &func);

  int size() const;

  virtual string getInfo();  
  virtual string getTypeName(int=0);

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static string typeName(int);
  static Persistent* maker();

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


//////////
// PIO support
template <class T> 
Persistent* BrickAttrib<T>::maker(){
  return new BrickAttrib<T>(0);
}

template <class T>
string BrickAttrib<T>::typeName(int n){
  ASSERTRANGE(n, 0, 2);
  static string t1name    = findTypeName((T*)0);
  static string className = string("BrickAttrib<") + t1name +">";
  
  switch (n){
  case 1:
    return t1name;
  default:
    return className;
  }
}

template <class T> 
PersistentTypeID BrickAttrib<T>::type_id(BrickAttrib<T>::typeName(0), 
					 FlatAttrib<T>::typeName(0), 
					 BrickAttrib<T>::maker);


#define BRICKATTRIB_VERSION 1
template <class T> 
void BrickAttrib<T>::io(Piostream& stream){
  stream.begin_class(typeName(0).c_str(), BRICKATTRIB_VERSION);
  
  // -- base class PIO
  FlatAttrib<T>::io(stream);
  
  Pio(stream, xbrickcount);
  Pio(stream, ybrickcount);
  Pio(stream, zbrickcount);

  stream.end_class();
}


template <class T> void
BrickAttrib<T>::update_brick_counts()
{
#if BITBOUND
  xbrickcount = nx_ / XBRICKSIZE;
  if (nx_ % XBRICKSIZE) xbrickcount++;

  ybrickcount = ny_ / YBRICKSIZE;
  if (ny_ % YBRICKSIZE) ybrickcount++;

  zbrickcount = nz_ / ZBRICKSIZE;
  if (nz_ % ZBRICKSIZE) zbrickcount++;
#else
  xbrickcount = nx_ >> XBRICKBITS;
  if (nx_ & XBRICKSPACE) xbrickcount++;

  ybrickcount = ny_ >> YBRICKBITS;
  if (ny_ & YBRICKSPACE) ybrickcount++;

  zbrickcount = nz_ >> ZBRICKBITS;
  if (nz_ & ZBRICKSPACE) zbrickcount++;
#endif  
}

template <class T>
BrickAttrib<T>::BrickAttrib(int x) :
  FlatAttrib<T>(x)
{
  update_brick_counts();
#if BITBOUND
  data_.resize(xbrickcount * XBRICKSIZE);
#else
  data_.resize(xbrickcount << XBRICKBITS);
#endif
}

template <class T>
BrickAttrib<T>::BrickAttrib(int x, int y) :
  FlatAttrib<T>(x, y)
{
  update_brick_counts();
#if BITBOUND
  data_.resize(xbrickcount * XBRICKSIZE *
		ybrickcount * YBRICKSIZE);
#else
  data_.resize((xbrickcount * ybrickcount) <<
		(XBRICKBITS + YBRICKBITS));
#endif
}

template <class T>
BrickAttrib<T>::BrickAttrib(int x, int y, int z) :
  FlatAttrib<T>(x, y, z)
{
  update_brick_counts();
#if BITBOUND
  data_.resize(xbrickcount * XBRICKSIZE *
		ybrickcount * YBRICKSIZE *
		zbrickcount * XBRICKSIZE);
#else
  data_.resize((xbrickcount * ybrickcount * zbrickcount) <<
	      (XBRICKBITS + YBRICKBITS + ZBRICKBITS));
#endif
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
  ASSERTEQ(dim_, 1);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  return data_[ix];
}

template <class T> T &
BrickAttrib<T>::fget2(int ix, int iy)
{
  ASSERTEQ(dim_, 2);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  return data_[linearize(ix, iy)];
}

template <class T> T &
BrickAttrib<T>::fget3(int ix, int iy, int iz)
{
  ASSERTEQ(dim_, 3);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  CHECKARRAYBOUNDS(iz, 0, nz_);
  return data_[linearize(ix, iy, iz)];
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
  ASSERTEQ(dim_, 1);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  data_[ix] = val;
}

template <class T> void
BrickAttrib<T>::fset2(int ix, int iy, const T& val)
{
  ASSERTEQ(dim_, 2);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  data_[linearize(ix, iy)] = val;
}


template <class T> void
BrickAttrib<T>::fset3(int ix, int iy, int iz, const T& val)
{
  ASSERTEQ(dim_, 3);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  CHECKARRAYBOUNDS(iz, 0, nz_);
  data_[linearize(ix, iy, iz)] = val;
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
BrickAttrib<T>::size() const
{
  switch (dim_)
    {
    case 3:
      return nx_ * ny_ * nz_;
    case 2:
      return nx_ * ny_;
    case 1:
      return nx_;
    default:
      return 0;
    }
}


template <class T> int
BrickAttrib<T>::iterate(AttribFunctor<T> &func)
{
  if (dim_ == 3)
    {
      for (int i = 0; i < nz_; i++)
	{
	  for (int j = 0; j < ny_; j++)
	    {
	      for (int k = 0; k < nx_; k++)
		{
		  func(fget3(k, j, i));
		}
	    }
	}
      return size();
    }
  else if (dim_ == 2)
    {
      for (int i = 0; i < ny_; i++)
	{
	  for (int j = 0; j < nx_; j++)
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
    "Name = " << name_ << endl <<
    "Type = BrickAttrib" << endl <<
    "Dim = " << dim_ << ": " << nx_ << ' ' << ny_ << ' ' << nz_ << endl <<
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
  vector<T>::iterator itr = data_.begin();
  int i = 0;
  for(;itr!=data_.end() && i < 200; itr++, i++) {
    retval << *itr << " ";
  }
  if (itr != data_.end()) { retval << "..."; }
  retval << endl;
  return retval.str();
}

template <class T> string
BrickAttrib<T>::getTypeName(int n){
  return typeName(n);
}

} // End namespace SCIRun



#endif  // SCI_project_BrickAttrib_h
