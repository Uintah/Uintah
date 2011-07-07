#ifndef __SPATIALOPS_MATERIAL_H__
#define __SPATIALOPS_MATERIAL_H__

#include <Core/Grid/Material.h>


namespace Uintah {
  class Burn;
  class SpatialOpsMaterial : public Material {
  public:
    SpatialOpsMaterial();
    
    ~SpatialOpsMaterial();
    
    Burn* getBurnModel() {
      return 0;
    }
  private:

    // Prevent copying of this class
    // copy constructor
    SpatialOpsMaterial(const SpatialOpsMaterial &spatialopsmm);
    SpatialOpsMaterial& operator=(const SpatialOpsMaterial &spatialopsmm);
  };

} // End namespace Uintah

#endif // __SPATIALOPS_MATERIAL_H__
