
#ifndef Datatypes_Field_h
#define Datatypes_Field_h

#include <Core/Datatypes/PropertyManager.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/MeshBase.h>
#include <Core/Containers/LockingHandle.h>

namespace SCIRun {

class  SCICORESHARE Field: public PropertyManager {

public:

  // GROUP: Constructors/Destructor
  //////////
  //
  virtual ~Field();

  // Required virtual functions
  virtual const MeshBase& get_mesh() const = 0;

  // Required interfaces
  virtual InterpolateToScalar* query_interpolate_to_scalar() const = 0;

};

typedef LockingHandle<Field> FieldHandle;

} // end namespace SCIRun

#endif // Datatypes_Field_h
















