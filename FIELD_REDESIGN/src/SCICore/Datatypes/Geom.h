//  Geom.h - Describes an entity in space -- abstract base class
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_Geom_h
#define SCI_project_Geom_h 1

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

#include <vector>
#include <string>


namespace SCICore {
namespace Datatypes{

using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class Geom;
typedef LockingHandle<Geom> GeomHandle;


// The bounding box
struct Bbox{
  Point min;
  Point max;
};

// The 4 ints indicate the 4 nodes that make up an element
struct Elem{
  int n[4];
};

class SCICORESHARE Geom:public Datatype{  
public:
  
  virtual ~Geom(){ };
  virtual Bbox& getBbox() {return bbox;};
  virtual void setBbox(const Bbox& ibbox) {bbox = ibbox;};
  virtual string get_info() = 0;
  
  inline void set_name(string iname) {name=iname;};
  inline string get_name() {return name;};
  
  // ...
protected:
  Bbox bbox;
  string name;
  Point bmin, bmax;
  Vector diagnal;
  
};


} // end namespace Datatypes
} // end namespace SCICore
  

#endif
