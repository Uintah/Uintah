//  Geom.h - Describes an entity in space -- abstract base class
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_Geom_h
#define SCI_project_Geom_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Persistent/Persistent.h>

#include <vector>
#include <string>
#include <iostream>

namespace SCIRun {

using std::vector;
using std::string;

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
  virtual ~Geom();

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

  //////////
  // Returns type name:
  // 0 - the class name
  // n!=0 - name of the n-th parameter (for templatazed types)
  virtual string getTypeName(int n) = 0;

  inline void setName(string iname) { name_ = iname; };
  inline string getName() { return name_; };

  /////////
  // Casts down to handle to attribute of specific type.
  // Returns empty handle if it was not successeful cast
  template <class T> LockingHandle<T> downcast(T*) {
    T* rep = dynamic_cast<T *>(this);
    return LockingHandle<T>(rep);
  }

  //////////
  // Persistent IO
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static string typeName(int);

  // ...
protected:

  //////////
  // Compute the bounding box, set has_bbox to true
  virtual bool computeBoundingBox() = 0;

  BBox bbox_;
  string name_;
};

} // End namespace SCIRun
  

#endif
