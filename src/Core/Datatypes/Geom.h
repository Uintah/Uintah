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
#include <SCICore/Datatypes/Attrib.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/BBox.h>

#include <vector>
#include <string>


namespace SCICore {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::BBox;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Math::Max;

class Geom;
typedef LockingHandle<Geom> GeomHandle;


//////////
// Where the attribute lives in the geometry
enum elem_t{
  NODE,
  EDGE,
  FACE,
  ELEM
};


class SCICORESHARE Geom : public Datatype {
public:

  Geom();
  virtual ~Geom() {};

  //////////
  // return the bounding box, if it is not allready computed 
  virtual bool getBoundingBox(BBox&);

  //////////
  // Compute the longest dimension
  bool longestDimension(double&);

  //////////
  // Return the diagonal
  bool getDiagonal(Vector&);
  
  //////////
  // Return a string describing this geometry.
  virtual string getInfo() = 0;

  inline void setName(string iname) { d_name = iname; };
  inline string getName() { return d_name; };
  
  // ...
protected:

  //////////
  // Compute the bounding box, set has_bbox to true
  virtual bool computeBoundingBox() = 0;

  BBox d_bbox;
  string d_name;
};

} // end namespace Datatypes
} // end namespace SCICore
  

#endif
