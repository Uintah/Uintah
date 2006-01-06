#ifndef __ARCHES_MATERIAL_H__
#define __ARCHES_MATERIAL_H__

#include <Packages/Uintah/Core/Grid/Material.h>


namespace Uintah {

      
/**************************************
     
CLASS
   ArchesMaterial

   Short description...

GENERAL INFORMATION

   ArchesMaterial.h

   Rajesh Rawat
   Department of Chemical and Fuels Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   Copyright (C) 2000 SCI Group

KEYWORDS
   MPM_Material

DESCRIPTION
   Long description...

WARNING

****************************************/
  class Burn;
  class ArchesMaterial : public Material {
  public:
    ArchesMaterial();
    
    ~ArchesMaterial();
    
    Burn* getBurnModel() {
      return 0;
    }
  private:

    // Prevent copying of this class
    // copy constructor
    ArchesMaterial(const ArchesMaterial &archesmm);
    ArchesMaterial& operator=(const ArchesMaterial &archesmm);
  };

} // End namespace Uintah

#endif // __ARCHES_MATERIAL_H__
