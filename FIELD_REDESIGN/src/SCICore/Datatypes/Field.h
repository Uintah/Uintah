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

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Datatypes/FieldInterface.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Datatypes/Geom.h>
#include <SCICore/Datatypes/Attrib.h>
#include <iostream>

#include <vector>
#include <string>

namespace SCICore{
namespace Datatypes{

using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;
using SCICore::Containers::LockingHandle;
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

//////////
// Where the attribute lives in the geometry
enum elem_t{
  NODAL,
  EDGE,
  FACE,
  ELEM
};


class SCICORESHARE Field:public Datatype{
public:
  //////////
  // Constructor
  Field();

  //////////
  // Virtual destructor 
  virtual ~Field() { };

  //////////
  // If geom is set, set its name
  inline void set_geom_name(string iname){if(geom.get_rep())
    geom->set_name(iname);};

  //////////
  // If attrib is set, set its name
  inline void set_attrib_name(string iname){if(attrib.get_rep())
    attrib->set_name(iname);};

  //////////
  // Return geometry
  inline GeomHandle get_geom(){return geom;};

  //////////
  // Return the attribute
  inline AttribHandle get_attrib(){return attrib;};

  //////////
  // Test to see if this field includes (is derived from)
  // the given interface. As convention, the input string should be in
  // all lowercase and be exactly the same as the FieldInterface's
  // name.  For example:
  //
  // SInterpolate *inter = some_field.query_interface("sinterpolate");
  //
  // Returns NULL if the given interface is not available for the field.
  virtual FieldInterface* query_interface(const string&);

  /////////
  // The current status of the field
  status_t status;
protected:

  /////////
  // The geometry and attribute associated with this field
  GeomHandle geom;
  AttribHandle attrib;

  /////////
  // Where the attributes live relative to the geometry.  Used only
  // for unstructured geometries.
  elem_t elem_type;
};



} // end namespace Datatypes
} // end namespace SCICore


#endif


