//  FieldWrapper.h - provides a wrapper for passing fields between the
//  register module and the domain manager module.
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute
//
//  TODO:
//  - Test the use of FieldHandle rather than SFieldHandle, 
//  VFieldHandle, ...
//  - (or) fill in to accomodate other field types.


#ifndef SCI_project_FieldWrapper_h
#define SCI_project_FieldWrapper_h 1

#include <SCICore/Datatypes/SField.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Datatypes/Geom.h>
#include <SCICore/Datatypes/Attrib.h>



#include <vector>
#include <string>

namespace SCICore{
namespace Datatypes{

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class FieldWrapper;
typedef LockingHandle<FieldWrapper> FieldWrapperHandle;

enum field_t{
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
  FieldWrapper(const GeomHandle&, status_t);
  FieldWrapper(const AttribHandle&, status_t);
  FieldWrapper(const SFieldHandle&, status_t);

  inline GeomHandle get_geom(){return mygeom;};
  inline AttribHandle get_attrib(){return myattrib;};
  inline SFieldHandle get_sfield(){return mysfield;};
  
  inline field_t get_field_type(){return fieldtype;};
  inline status_t get_status_type(){return statustype;};
  //inline void set_field_type(field_t inval){fieldtype=inval;};
  //inline void set_status_type(status_t inval){statustype=inval;};
  
  
  // Persistent representation...
  virtual void io(Piostream&) { };
  static PersistentTypeID type_id;
  
private:
  GeomHandle mygeom;
  AttribHandle myattrib;
  SFieldHandle mysfield;
  field_t fieldtype;
  status_t statustype;
};
  

} // end namesapace Datatypes
} // end namesapace SCICore


#endif
