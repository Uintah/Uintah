//  Accel3Attrib.h - scalar attribute stored as a flat array
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_Accel3Attrib_h
#define SCI_project_Accel3Attrib_h 1

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

template <class T> class Accel3Attrib : public FlatAttrib<T>
{
public:
  
  //////////
  // Constructors
  Accel3Attrib();
  Accel3Attrib(int, int, int);
  Accel3Attrib(const Accel3Attrib<T> &copy);

  //////////
  // Destructor
  virtual ~Accel3Attrib();
  
  //////////
  // return the value at the given position
  T& get3(int, int, int);

  T& set3(int, int, int, T& val);

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
  vector<vector<T *> > accel;
};



template <class T>
void
Accel3Attrib<T>::create_accel_structure()
{
  accel.resize(nz);
  for (int i=0; i < nz; i++)
    {
      accel[i].resize(ny);
      for (int j=0; j < ny; j++)
	{
	  accel[i][j] = &data[i*nx*ny + j*nx];
	}
    }
}


template <class T>
Accel3Attrib<T>::Accel3Attrib()
  : FlatAttrib<T>()
{
  dim = 3;
}

template <class T>
Accel3Attrib<T>::Accel3Attrib(int ix, int iy, int iz)
  : FlatAttrib<T>(ix, iy, iz)
{
  create_accel_structure();
}


template <class T>
Accel3Attrib<T>::Accel3Attrib(const Accel3Attrib<T>& copy)
  : FlatAttrib<T>(copy)
{
  create_accel_structure();
}


template <class T>
T&
Accel3Attrib<T>::get3(int ix, int iy, int iz)
{
  return accel[iz][iy][ix];
}


template <class T>
T&
Accel3Attrib<T>::set3(int ix, int iy, int iz, T& val)
{
  return accel[iz][iy][ix] = val;
}


template <class T>
void
Accel3Attrib<T>::resize(int)
{
  throw DimensionMismatch(1, 3);
}


template <class T>
void
Accel3Attrib<T>::resize(int, int)
{
  throw DimensionMismatch(2, 3);
}


template <class T>
void
Accel3Attrib<T>::resize(int x, int y, int z)
{
  FlatAttrib<T>::resize(x, y, z);
  create_accel_structure();
}



template <class T>
PersistentTypeID Accel3Attrib<T>::type_id("Accel3Attrib", "Datatype", 0);


template <class T>
string
Accel3Attrib<T>::get_info()
{
  ostringstream retval;
  retval <<
    "Name = " << name << '\n' <<
    "Type = Accel3Attrib" << '\n' <<
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
Accel3Attrib<T>::io(Piostream&)
{
}

}  // end Datatypes
}  // end SCICore



#endif  // SCI_project_Accel3Attrib_h







