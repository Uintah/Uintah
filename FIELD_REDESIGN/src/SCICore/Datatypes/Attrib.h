// Attrib.h - the base attribute class.
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_Attrib_h
#define SCI_project_Attrib_h 1

#include <vector>
#include <string>

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>


namespace SCICore{
namespace Datatypes{

using namespace std;
using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

template <class T> class FlatSAttrib;
class Attrib;
typedef LockingHandle<Attrib> AttribHandle;

class Attrib:public Datatype 
{
public:
  //////////
  // Constructors, Destructor
  Attrib(int, int, int);
  Attrib(int, int);
  Attrib(int);
  Attrib();
  Attrib(const Attrib& copy);
  virtual ~Attrib() { };
  
  /////////
  // Cast to a FlatSAttrib and return
  template <class T> FlatSAttrib<T>* get_flatsattrib();
  
  /////////
  // set (and get) the name of the attribute
  void set_name(std::string iname){name=iname; };
  std::string get_name(){return name;};

  /////////
  //
  virtual string get_info() =0;
  
  /////////
  // resize the geometry
  virtual void resize(int x, int y, int z) = 0;
  virtual void resize(int x, int y) = 0;
  virtual void resize(int x) = 0;
  
protected:
  /////////
  // an identifying name
  std::string name;
  int nx, ny, nz;
  int dims_set;
};

template <class T> FlatSAttrib<T>* Attrib::get_flatsattrib(){
  return dynamic_cast<FlatSAttrib<T>*>(this);
}

}  // end Datatypes
}  // end SCICore

#endif



