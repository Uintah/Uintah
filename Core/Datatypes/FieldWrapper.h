//  FieldWrapper.h - provides a wrapper for passing fields between the
//  register module and the domain manager module.
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute
//  TODO:
//  - Test the use of FieldHandle rather than SFieldHandle, 
//  VFieldHandle, ...
//  - (or) fill in to accomodate other field types.


#ifndef SCI_project_FieldWrapper_h
#define SCI_project_FieldWrapper_h 1

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Geom.h>
#include <Core/Datatypes/Attrib.h>



#include <vector>
#include <string>

namespace SCIRun {


class FieldWrapper;
typedef LockingHandle<FieldWrapper> FieldWrapperHandle;

enum field_t {
  FIELD = 1,
  SFIELD,
  TFIELD,
  VFIELD,
  GEOM,
  ATTRIB
};

class SCICORESHARE FieldWrapper:public Datatype{

public:
  /////////
  // Constructors
  FieldWrapper(const GeomHandle&);
  FieldWrapper(const AttribHandle&);
  FieldWrapper(const FieldHandle&);

  inline GeomHandle get_geom(){return geom_;};
  inline AttribHandle get_attrib(){return attrib_;};
  inline FieldHandle get_field(){return field_;};
  
  inline field_t get_field_type(){return fieldtype;};
  //inline status_t get_status_type(){return statustype;};
  //inline void set_field_type(field_t inval){fieldtype=inval;};
  //inline void set_status_type(status_t inval){statustype=inval;};
  
  
  // Persistent representation...
  virtual void io(Piostream&) { };
  static PersistentTypeID type_id;
  
private:
  GeomHandle geom_;
  AttribHandle attrib_;
  FieldHandle field_;
  field_t fieldtype;
  //status_t statustype;
};
  

} // End namespace SCIRun


#endif
