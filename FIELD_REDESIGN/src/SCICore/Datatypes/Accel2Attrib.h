//  Accel2Attrib.h - scalar attribute stored as a flat array
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_Accel2Attrib_h
#define SCI_project_Accel2Attrib_h 1

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

template <class T> class Accel2Attrib : public FlatAttrib<T>
{
public:
  
  //////////
  // Constructors
  Accel2Attrib();
  Accel2Attrib(int, int);
  Accel2Attrib(const Accel2Attrib<T> &copy);

  //////////
  // Destructor
  virtual ~Accel2Attrib();
  
  virtual void get2(T &result, int x, int y);

  virtual T &get2(int x, int y);

  //////////
  // return the value at the given position
  T& fget2(int, int);


  virtual void set2(int, int, const T &val);

  void fset2(int, int, const T &val);

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
  vector<T *> accel;
};



template <class T>
void
Accel2Attrib<T>::create_accel_structure()
{
  accel.resize(ny);
  for (int i=0; i < ny; i++)
    {
      accel[i] = &data[i*nx];
    }
}


template <class T>
Accel2Attrib<T>::Accel2Attrib()
  : FlatAttrib<T>()
{
  dim = 2;
}

template <class T>
Accel2Attrib<T>::Accel2Attrib(int ix, int iy)
  : FlatAttrib<T>(ix, iy)
{
  create_accel_structure();
}


template <class T>
Accel2Attrib<T>::Accel2Attrib(const Accel2Attrib<T>& copy)
  : FlatAttrib<T>(copy)
{
  create_accel_structure();
}


template <class T>
T &
Accel2Attrib<T>::fget2(int ix, int iy)
{
  return accel[iy][ix];
}


template <class T> void
Accel2Attrib<T>::fset2(int x, int y, const T& val)
{
  accel[y][x] = val;
}


template <class T> void
Accel2Attrib<T>::set2(int x, int y, const T &val)
{
  fset2(x, y, val);
}


// Copy wrappers, no allocation of result.
template <class T> void
Accel2Attrib<T>::get2(T &result, int ix, int iy)
{
  result = fget2(ix, iy);
}


// Virtual wrappers for inline functions.
template <class T> T &
Accel2Attrib<T>::get2(int ix, int iy)
{
  return fget2(ix, iy);
}


template <class T>
void
Accel2Attrib<T>::resize(int)
{
  throw DimensionMismatch(1, 2);
}


template <class T>
void
Accel2Attrib<T>::resize(int x, int y)
{
  FlatAttrib<T>::resize(x, y);
  create_accel_structure();
}


template <class T>
void
Accel2Attrib<T>::resize(int, int, int)
{
  throw DimensionMismatch(3, 2);
}



template <class T>
PersistentTypeID Accel2Attrib<T>::type_id("Accel2Attrib", "Datatype", 0);


template <class T>
string
Accel2Attrib<T>::get_info()
{
  ostringstream retval;
  retval <<
    "Name = " << name << '\n' <<
    "Type = Accel2Attrib" << '\n' <<
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
Accel2Attrib<T>::io(Piostream&)
{
}

}  // end Datatypes
}  // end SCICore



#endif  // SCI_project_Accel2Attrib_h







