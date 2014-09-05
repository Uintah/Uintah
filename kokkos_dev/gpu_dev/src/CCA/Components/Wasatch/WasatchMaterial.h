#ifndef Wasatch_Material_h
#define Wasatch_Material_h

#include <Core/Grid/Material.h>
#include <Core/Grid/uintahshare.h>
namespace Uintah {

using namespace SCIRun;
/**
 *  \class  WasatchMaterial
 *  \author Tony Saad
 *  \date   December 2011
 *
 *  \brief Create a simple wasatch material.
 */
  
      class UINTAHSHARE WasatchMaterial : public Material {
      public:
         WasatchMaterial();         
         ~WasatchMaterial();
 
      private:
      };

} // End namespace Uintah

#endif // Wasatch_Material_h
