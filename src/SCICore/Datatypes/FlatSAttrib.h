//  FlatSAttrib.h - scalar attribute stored as a flat array
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_FlatSAttrib_h
#define SCI_project_FlatSAttrib_h 1

#include <vector>

#include <SCICore/Datatypes/SAttrib.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Exceptions/ArrayIndexOutOfBounds.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Util/DebugStream.h>
#include <sstream>


namespace SCICore{
namespace Datatypes{

using std::vector;
using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Exceptions::ArrayIndexOutOfBounds;
using SCICore::Math::Min;
using SCICore::Math::Max;
using std::ostringstream;
using SCICore::Util::DebugStream;

template <class T> class FlatSAttrib:public SAttrib 
{
public:
  
  //////////
  // Constructors
  FlatSAttrib();
  FlatSAttrib(const FlatSAttrib& copy);
  FlatSAttrib(int, int, int);
  FlatSAttrib(int, int);
  FlatSAttrib(int);
  
  //////////
  // Destructor
  ~FlatSAttrib();
  
  //////////
  // return the value at the given position
  T& operator[](int);
  T& grid(int, int, int);

  /////////
  // Return the min and max data values;
  virtual bool compute_minmax();

  //////////
  // Resize the attribute to the specified dimensions
  virtual void resize(int, int, int);

  virtual string get_info();  

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  
private:
  vector<T> data;
  static DebugStream dbg;
};

}
}

namespace SCICore{
namespace Datatypes{

template <class T> DebugStream FlatSAttrib<T>::dbg("FlatSAttrib", true);

template <class T> PersistentTypeID FlatSAttrib<T>::type_id("FlatSAttrib", "Datatype", 0);

template <class T> FlatSAttrib<T>::FlatSAttrib()
  :SAttrib(){
}

template <class T> FlatSAttrib<T>::FlatSAttrib(const FlatSAttrib& copy)
  :SAttrib(copy), data(copy.data){
}

template <class T> FlatSAttrib<T>::FlatSAttrib(int ix, int iy, int iz):
  SAttrib(ix, iy, iz){
  data.resize(ix*iy*iz);
}

template <class T> FlatSAttrib<T>::FlatSAttrib(int ix, int iy):
  SAttrib(ix, iy){
  data.resize(ix*iy);
}

template <class T> FlatSAttrib<T>::FlatSAttrib(int ix):
  SAttrib(ix){
  data.resize(ix);
}

template <class T> FlatSAttrib<T>::~FlatSAttrib(){
}

template <class T> T& FlatSAttrib<T>::operator[](int i){
  if(i > data.size() - 1){
    throw ArrayIndexOutOfBounds(data.size(), 0, i);
  }
  return data[i];
}

template <class T> T& FlatSAttrib<T>::grid(int ix, int iy, int iz){
  
  //  if(dims_set < 3){
  //  throw ArrayIndexOutOfBounds(ix, 0, nx);
  //}
  //else if(ix>=nx || ix<0){
  //  throw ArrayIndexOutOfBounds(ix, 0, nx);
  //}
  //else if(iy>=ny || iz>=nz || ix<0 || iy<0 || iz<0){
  // dbg << "grid("<<ix<<","<<iy<<","<<iz<<") == ["<<(iz*(nx*ny)+iy*(nx)+ix)<<"]"<<"=="<<data[iz*(nx*ny)+iy*(nx)+ix]<<endl;
  return data[iz*(nx*ny)+iy*(nx)+ix];  
}

template <class T> bool FlatSAttrib<T>::compute_minmax(){
  has_minmax = 1;
  if(data.empty()){
    min = 0;
    max = 0;
    return false;
  }
  else{
    vector<T>::iterator itr;
    T lmin = data[0];
    T lmax = lmin;
    for(itr = data.begin(); itr != data.end(); itr++){
      lmin = Min(lmin, *itr);
      lmax = Max(lmax, *itr);
    }
    min = (double) lmin;
    max = (double) lmax;
    return true;
  }
}

template <class T> void FlatSAttrib<T>::resize(int x, int y, int z){
  nx = x;
  ny = y;
  nz = z;
  dims_set = 3;
  data.resize(x*y*z);
}

template <class T> string FlatSAttrib<T>::get_info(){
  ostringstream retval;
  retval <<
    "Name = " << name << '\n' <<
    "Type = FlatSAttrib" << '\n' <<
    "Size = " << data.size() << '\n' <<
    "Data = ";
  vector<T>::iterator itr = data.begin();
  for(;itr!=data.end(); itr++){
    retval << *itr << " ";
  }
  retval << endl;
  return retval.str();
}

template <class T> void FlatSAttrib<T>::io(Piostream&){
}

}  // end Datatypes
}  // end SCICore



#endif



