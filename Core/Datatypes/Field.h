// Field.h - This is the base class from which all other fields are derived.
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
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

#include <functional>
#include <iostream>
#include <vector>
#include <string>

namespace SCIRun {

using std::vector;
using std::string;
using std::cerr;
using std::endl;

class Field;
typedef LockingHandle<Field> FieldHandle;

//////////
// The current status of the field
enum status_t{  
  OLD,
  NEW,
  SHARED
};


class SCICORESHARE Field:public Datatype{
public:
  //////////
  // Constructor
  Field();

  //////////
  // Virtual destructor 
  virtual ~Field();

  //////////
  // Return geometry
  virtual Geom* get_geom() = 0;
  
  //////////
  // Return the attribute
  virtual Attrib* get_attrib() = 0;


  inline Field * get_base() { return this; }

  //////////
  // Test to see if this field includes (is derived from)
  // the given interface. As convention, the input string should be in
  // all lowercase and be exactly the same as the FieldInterface's
  // name.  For example:
  // SInterpolate *inter = some_field.query_interface("sinterpolate");
  // Returns NULL if the given interface is not available for the field.
  template <class T> T* query_interface(T *);
  
  /////////
  // The current status of the field
  status_t status;

  /////////
  // Where the attributes live relative to the geometry.  Used only
  // for unstructured geometries.
  elem_t data_loc;

  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
protected:
};



} // End namespace SCIRun


#endif


