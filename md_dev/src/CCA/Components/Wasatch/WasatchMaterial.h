#ifndef Wasatch_Material_h
#define Wasatch_Material_h

#include <Core/Grid/Material.h>

namespace Uintah {

  /**
   *  \class  WasatchMaterial
   *  \author Tony Saad
   *  \date   December 2011
   *
   *  \brief Create a simple wasatch material.
   */

  class WasatchMaterial : public Material {
  public:
    WasatchMaterial(){}
    ~WasatchMaterial(){}
  private:
  };

} // End namespace Uintah

#endif // Wasatch_Material_h
