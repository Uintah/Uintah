#ifndef Datatypes_MeshBase_h
#define Datatypes_MeshBase_h

#include <Core/Datatypes/PropertyManager.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/LockingHandle.h>

namespace SCIRun {

class MeshBase : public PropertyManager {
public:
  virtual ~MeshBase();
  
  //! Required virtual functions
  virtual BBox get_bounding_box() const = 0;
  
  // Required interfaces


  //! Persistent I/O.
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int);
  //! All instantiable classes need to define this.
  virtual const string get_type_name(int n) const = 0;
};

typedef LockingHandle<MeshBase> MeshBaseHandle;

} // end namespace SCIRun

#endif // Datatypes_MeshBase_h
