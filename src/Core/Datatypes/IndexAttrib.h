//  IndexAttrib.h
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
//
//  Copyright (C) 2000 SCI Institute
//
//  Attribute containing a finite number of Index values.
//

#ifndef SCI_project_IndexAttrib_h
#define SCI_project_IndexAttrib_h 1

#include <vector>

#include <SCICore/Datatypes/AccelAttrib.h>


namespace SCICore{
namespace Datatypes{

using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Point;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

template <class T, class I, class A=AccelAttrib<I> > class IndexAttrib : public DiscreteAttrib<T>
{
public:
  IndexAttrib();
  IndexAttrib(int x);
  IndexAttrib(int x, int y);
  IndexAttrib(int x, int y, int z);
  
  IndexAttrib(const IndexAttrib& copy);

  virtual ~IndexAttrib();

  virtual void get1(T &result, int x);
  virtual void get2(T &result, int x, int y);
  virtual void get3(T &result, int x, int y, int z);

  virtual T &get1(int x);
  virtual T &get2(int x, int y);
  virtual T &get3(int x, int y, int z);

  T &fget1(int x);
  T &fget2(int x, int y);
  T &fget3(int x, int y, int z);
  

  virtual void iset1(int x, const I &val);
  virtual void iset2(int x, int y, const I &val);
  virtual void iset3(int x, int y, int z, const I &val);

  void fiset1(int x, const I &val);
  void fiset2(int x, int y, const I &val);
  void fiset3(int x, int y, int z, const I &val);

  virtual void tset(int x, const T &val);
  void ftset(int x, const T &val);



  // Implement begin()
  // Implement end()

  // Resize the attribute to the specified dimensions
  virtual void resize(int, int, int);
  virtual void resize(int, int);
  virtual void resize(int);

  virtual string getInfo();  

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  virtual int iterate(AttribFunctor<T> &func);

  //protected:

  //private:
  A iattrib;

  vector<T> index;
};


template <class T, class I, class A>
IndexAttrib<T, I, A>::IndexAttrib() :
  DiscreteAttrib<T>(),
  iattrib(),
  index(sizeof(I) * 256, I())
{
}

template <class T, class I, class A>
IndexAttrib<T, I, A>::IndexAttrib(int ix) :
  DiscreteAttrib<T>(ix),
  iattrib(ix),
  index(sizeof(I) * 256, I())
{
}

template <class T, class I, class A>
IndexAttrib<T, I, A>::IndexAttrib(int ix, int iy) :
  DiscreteAttrib<T>(ix, iy),
  iattrib(ix, iy),
  index(sizeof(I) * 256, I())
{
}

template <class T, class I, class A>
IndexAttrib<T, I, A>::IndexAttrib(int ix, int iy, int iz) :
  DiscreteAttrib<T>(ix, iy, iz),
  iattrib(ix, iy, iz),
  index(sizeof(I) * 256, I())
{
}

template <class T, class I, class A>
IndexAttrib<T, I, A>::IndexAttrib(const IndexAttrib& copy) :
  DiscreteAttrib<T>(copy), iattrib(copy.iattrib), index(copy.index)
{
}


template <class T, class I, class A>
IndexAttrib<T, I, A>::~IndexAttrib()
{
}


template <class T, class I, class A> T &
IndexAttrib<T, I, A>::fget1(int x)
{
  return index[iattrib.fget1(x)];
}


template <class T, class I, class A> T &
IndexAttrib<T, I, A>::fget2(int x, int y)
{
  return index[iattrib.fget2(x, y)];
}


template <class T, class I, class A> T &
IndexAttrib<T, I, A>::fget3(int x, int y, int z)
{
  return index[iattrib.fget3(x, y, z)];
}


// Copy wrappers, no allocation of result.
template <class T, class I, class A> void
IndexAttrib<T, I, A>::get1(T &result, int ix)
{
  result = fget1(ix);
}

template <class T, class I, class A> void
IndexAttrib<T, I, A>::get2(T &result, int ix, int iy)
{
  result = fget2(ix, iy);
}

template <class T, class I, class A> void
IndexAttrib<T, I, A>::get3(T &result, int ix, int iy, int iz)
{
  result = fget3(ix, iy, iz);
}


// Virtual wrappers for inline functions.
template <class T, class I, class A> T &
IndexAttrib<T, I, A>::get1(int ix)
{
  return fget1(ix);
}

template <class T, class I, class A> T &
IndexAttrib<T, I, A>::get2(int ix, int iy)
{
  return fget2(ix, iy);
}

template <class T, class I, class A> T &
IndexAttrib<T, I, A>::get3(int ix, int iy, int iz)
{
  return fget3(ix, iy, iz);
}



template <class T, class I, class A> void
IndexAttrib<T, I, A>::fiset1(int x, const I &val)
{
  iattrib.fset1(x, val);
}


template <class T, class I, class A> void
IndexAttrib<T, I, A>::fiset2(int x, int y, const I &val)
{
  iattrib.fset2(x, y, val);
}


template <class T, class I, class A> void
IndexAttrib<T, I, A>::fiset3(int x, int y, int z, const I &val)
{
  iattrib.fset3(x, y, z, val);
}


// Generic setters for Index type
template <class T, class I, class A> void
IndexAttrib<T, I, A>::iset1(int x, const I &val)
{
  fiset1(x, val);
}

template <class T, class I, class A> void
IndexAttrib<T, I, A>::iset2(int x, int y, const I &val)
{
  fiset2(x, y, val);
}

template <class T, class I, class A> void
IndexAttrib<T, I, A>::iset3(int x, int y, int z, const I &val)
{
  fiset3(x, y, z, val);
}


template <class T, class I, class A> void
IndexAttrib<T, I, A>::ftset(int x, const T &val)
{
  index[x] = val;
}


template <class T, class I, class A> void
IndexAttrib<T, I, A>::tset(int x, const T &val)
{
  ftset(x, val);
}




// template <class T, class I, class A> bool IndexAttrib<T, I, A>::compute_minmax(){
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

template <class T, class I, class A> void
IndexAttrib<T, I, A>::resize(int x, int y, int z)
{
  DiscreteAttrib<T>::resize(x, y, z);
  iattrib.resize(x, y, z);
}


template <class T, class I, class A> void
IndexAttrib<T, I, A>::resize(int x, int y)
{
  DiscreteAttrib<T>::resize(x, y);
  iattrib.resize(x, y);
}


template <class T, class I, class A> void
IndexAttrib<T, I, A>::resize(int x)
{
  DiscreteAttrib<T>::resize(x);
  iattrib.resize(x);
}


template <class T, class I, class A> int
IndexAttrib<T, I, A>::iterate(AttribFunctor<T> &func)
{
  vector<T>::iterator itr = index.begin();
  while (itr != index.end())
    {
      func(*itr++);
    }
  return index.size();
}


template <class T, class I, class A> PersistentTypeID IndexAttrib<T, I, A>::type_id("IndexAttrib", "Datatype", 0);


template <class T, class I, class A> string
IndexAttrib<T, I, A>::getInfo()
{
  ostringstream retval;
  retval <<
    "Name = " << d_name << endl <<
    "Type = IndexAttrib" << endl <<
    "Dim = " << d_dim << ": " << d_nx << ' ' << d_ny << ' ' << d_nz << endl <<
    "Size = " << size() << endl <<
    "Datasize = " << index.size() << endl <<
    "Data = ";
  vector<T>::iterator itr = index.begin();
  int i = 0;
  for(;itr!=index.end() && i < 1000; itr++, i++)  {
    retval << *itr << " ";
  }
  if (itr != index.end()) { retval << "..."; }
  retval << endl;
  retval << 
    "SubAttrib =" << endl << iattrib.getInfo();
  return retval.str();
}

template <class T, class I, class A> void
IndexAttrib<T, I, A>::io(Piostream&)
{
}


}  // end Datatypes
}  // end SCICore

#endif



