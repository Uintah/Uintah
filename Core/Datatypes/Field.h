/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


#ifndef Datatypes_Field_h
#define Datatypes_Field_h

#include <Core/Datatypes/PropertyManager.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/MeshBase.h>
#include <Core/Containers/LockingHandle.h>

namespace SCIRun {

class  SCICORESHARE Field: public PropertyManager {

public:
  //! Possible data associations.
  enum data_location{
    NODE,
    EDGE,
    FACE,
    CELL,
    NONE
  };


  Field(data_location at = NONE);
  virtual ~Field();

  data_location data_at() const { return data_at_; }
  //! Required virtual functions
  virtual MeshBaseHandle mesh() const = 0;

  //! Required interfaces
  virtual InterpBase* query_interpolate() const = 0;
//  virtual InterpolateToScalar* query_interpolate_to_scalar() const = 0;

  //! Persistent I/O.
  virtual void io(Piostream &stream);
  static  PersistentTypeID type_id;
  //! All instantiable classes need to define this.
  virtual const string get_type_name(int n = -1) const = 0;
  virtual bool is_scalar() const = 0;

protected:

  //! Where data is associated.
  data_location           data_at_;
};

typedef LockingHandle<Field> FieldHandle;

} // end namespace SCIRun

#endif // Datatypes_Field_h
















