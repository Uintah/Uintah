//  FieldWrapper.cc - provides a wrapper for passing fields between the
//  register module and the domain manager module.
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/FieldWrapper.h>


namespace SCICore{
namespace Datatypes{

PersistentTypeID FieldWrapper::type_id("FieldWrapper", "Datatype", 0);


  FieldWrapper::FieldWrapper(const GeomHandle &handle, status_t intype):
  statustype(intype), fieldtype(GEOM){
  d_geom = handle;
}

FieldWrapper::FieldWrapper(const AttribHandle &handle, status_t intype):
  statustype(intype), fieldtype(ATTRIB){
  d_attrib = handle;
}

FieldWrapper::FieldWrapper(const FieldHandle &handle, status_t intype):
  statustype(intype), fieldtype(FIELD) {
  d_field = handle;
}


}  // end Datatypes
}  // end SCICore
