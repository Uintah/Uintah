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

  virtual string getInfo();  

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

protected:
  void create_accel_structure();

  // Accel structure.
  vector<vector<T *> > d_accel3;
  vector<T *> d_accel2;
};



template <class T> PersistentTypeID AccelAttrib<T>::type_id("AccelAttrib", "Datatype", 0);


template <class T>
void
AccelAttrib<T>::create_accel_structure()
{
  d_data.reserve(d_data.size());
  if (d_dim == 3)
    {
      d_accel3.resize(d_nz);
      for (int i=0; i < d_nz; i++)
	{
	  d_accel3[i].resize(d_ny);
	  for (int j=0; j < d_ny; j++)
	    {
	      d_accel3[i][j] = &(d_data[i*d_nx*d_ny + j*d_nx]);
	    }
	}
    }
  else if (d_dim == 2)
    {
      d_accel2.resize(d_ny);
      for (int i=0; i < d_ny; i++)
	{
	  d_accel2[i] = &(d_data[i*d_nx]);
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
  ASSERTEQ(d_dim, 1);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  return d_data[ix];
}

template <class T> T &
AccelAttrib<T>::fget2(int ix, int iy)
{
  ASSERTEQ(d_dim, 2);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  return d_accel2[iy][ix];  
}

template <class T> T &
AccelAttrib<T>::fget3(int ix, int iy, int iz)
{
  ASSERTEQ(d_dim, 3);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  CHECKARRAYBOUNDS(iz, 0, d_nz);
  return d_accel3[iz][iy][ix];  
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
  ASSERTEQ(d_dim, 1);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  d_data[ix] = val;
}


template <class T> void
AccelAttrib<T>::fset2(int ix, int iy, const T& val)
{
  ASSERTEQ(d_dim, 2);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  d_accel2[iy][ix] = val;
}


template <class T> void
AccelAttrib<T>::fset3(int ix, int iy, int iz, const T& val)
{
  ASSERTEQ(d_dim, 3);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  CHECKARRAYBOUNDS(iz, 0, d_nz);
  d_accel3[iz][iy][ix] = val;
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


template <class T> string AccelAttrib<T>::getInfo() {
  ostringstream retval;
  retval <<
    "Name = " << d_name << endl <<
    "Type = AccelAttrib" << endl <<
    "Dim = " << d_dim << ": " << d_nx << ' ' << d_ny << ' ' << d_nz << endl <<
    "Size = " << size() << endl;
#if 1
  retval << "Data = ";
  vector<T>::iterator itr = d_data.begin();
  int i = 0;
  for(;itr!=d_data.end() && i < 1000; itr++, i++) {
    retval << *itr << " ";
  }
  if (itr != d_data.end()) { retval << "..."; }
  retval << endl;
#else
  for (int k = 0; k < d_nz; k++)
    {
      for (int j = 0; j < d_nz; j++)
	{
	  retval << "  " << &(d_data[k * d_nx*d_ny + j * d_nx]);
	}
      retval << endl;
    }
  retval << endl;
#endif
  return retval.str();
}


template<>
string
AccelAttrib<unsigned char>::getInfo()
{
  ostringstream retval;
  retval <<
    "Name = " << d_name << endl <<
    "Type = AccelAttrib" << endl <<
    "Dim = " << d_dim << ": " << d_nx << ' ' << d_ny << ' ' << d_nz << endl <<
    "Size = " << size() << endl;
#if 1
  retval << "Data = ";
  vector<unsigned char>::iterator itr = d_data.begin();
  int i = 0;
  for(;itr!=d_data.end() && i < 1000; itr++, i++) {
    retval << (int)(*itr) << " ";
  }
  if (itr != d_data.end()) { retval << "..."; }
  retval << endl;
#else
  for (int k = 0; k < d_nz; k++)
    {
      for (int j = 0; j < d_nz; j++)
	{
	  retval << "  " << (int)(&(d_data[k * d_nx*d_ny + j * d_nx]));
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



