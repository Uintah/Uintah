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
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/Mesh.h>
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
  virtual Field *clone() const = 0;
  
 
  data_location data_at() const { return data_at_; }
  virtual const TypeDescription *data_at_type_description() = 0;

  //! Required virtual functions
  virtual MeshHandle mesh() const = 0;
  virtual void mesh_detach() = 0;
  virtual const TypeDescription* get_type_description() const = 0; 
  

  //! Required interfaces
  virtual ScalarFieldInterface* query_scalar_interface() const { return 0; }
  virtual VectorFieldInterface* query_vector_interface() const { return 0; }
  virtual TensorFieldInterface* query_tensor_interface() const { return 0; }

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
















