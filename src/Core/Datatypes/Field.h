// Field.h - This is the base class from which all other fields are derived.
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_Field_h
#define SCI_project_Field_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Datatypes/Geom.h>
#include <Core/Datatypes/Attrib.h>
#include <Core/Containers/Array1.h>

#include <functional>
#include <iostream>
#include <vector>
#include <string>
#include <map>

namespace SCIRun{

using std::vector;
using std::string;
using std::map;

class Field;
typedef LockingHandle<Field>      FieldHandle;
typedef map<string, AttribHandle> AttribMap;
typedef Array1<AttribHandle>      AttribVector;

class SCICORESHARE Field: public Datatype{

public:

  // GROUP: Constructors/Destructor
  //////////
  //
  Field();
  Field(const Field&);
  virtual ~Field();

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  // GROUP: Member functions to manipulate attributes and geometries
  //////////
  // 

  //////////
  // Total number of registred attributes
  inline int getNumAttribs() const{
    return d_attribHandles.size();
  }

  //////////
  // Returns handle to the current active attribute
  virtual const AttribHandle getAttrib() const;
  
  //////////
  // set attribute with the name supplied as active one
  void setCurrAttrib(string aName);
  
  //////////
  // Return attribute with the name supplied
  const AttribHandle getAttrib(string aName) const;
  
  //////////
  // Returns non-const handle to share the attribute for writing
  virtual AttribHandle shareAttrib(string aName);

  //////////
  // Returns handle to the geometry
  virtual const GeomHandle getGeom() const;

  //////////
  // Adds attribute to the field and sets current attribute name to its name
  void addAttribute(const AttribHandle&);
  
  //////////
  // Removes attribute with particular order number from the field
  void removeAttribute(string);
  
  //////////
  // Adds geometry to the field
  // Returns false if the geometry was not registred
  // TODO: checking needed if the geometry is in correspondence with the field
  bool setGeometry(GeomHandle);
  
  //////////
  // Test to see if this field includes (is derived from)
  // the given interface. As convention, the input string should be in
  // all lowercase and be exactly the same as the FieldInterface's
  // name.  For example:
  //
  // SInterpolate *inter = some_field.query_interface("sinterpolate");
  //
  // Returns NULL if the given interface is not available for the field.
  template <class T> T* query_interface(T *);
  
  /////////
  // Casts down to handle to attribute of specific type.
  // Returns empty handle if it was not successeful cast
  template <class T> LockingHandle<T> downcast(T*) {
    T* rep = dynamic_cast<T *>(this);
    return LockingHandle<T>(rep);
  }

protected:
  AttribMap       d_attribHandles;
  GeomHandle      d_geomHandle;
  string          d_currAttrib;
};

template <class T> T* Field::query_interface(T *)
{
  return dynamic_cast<T*>(this);
}

} // end namespace SCIRun

#endif


