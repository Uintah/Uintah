#ifndef Datatypes_MeshBase_h
#define Datatypes_MeshBase_h

#include <Core/Datatypes/PropertyManager.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

class MeshBase : public PropertyManager {
public:
  virtual ~MeshBase();
  
  // Required virtual functions
  virtual BBox get_bounding_box() const = 0;
  
  // Required interfaces

};

}

#endif // Datatypes_MeshBase_h
