//  FieldWrapper.cc - provides a wrapper for passing fields between the
//  register module and the domain manager module.
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/FieldWrapper.h>


namespace SCIRun {

PersistentTypeID FieldWrapper::type_id("FieldWrapper", "Datatype", 0);


FieldWrapper::FieldWrapper(const GeomHandle &handle):
  fieldtype(GEOM){
  geom_ = handle;
}

FieldWrapper::FieldWrapper(const AttribHandle &handle):
  fieldtype(ATTRIB){
  attrib_ = handle;
}

FieldWrapper::FieldWrapper(const FieldHandle &handle):
  fieldtype(FIELD) {
  field_ = handle;
}


} // End namespace SCIRun
