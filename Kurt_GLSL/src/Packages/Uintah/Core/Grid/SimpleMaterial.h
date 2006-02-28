#ifndef Packages_Uintah_Core_Grid_SimpleMaterial_h
#define Packages_Uintah_Core_Grid_SimpleMaterial_h

#include <Packages/Uintah/Core/Grid/Material.h>
namespace Uintah {

using namespace SCIRun;

/**************************************
     
CLASS
   SimpleMaterial

   Short description...

GENERAL INFORMATION

   SimpleMaterial.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   Copyright (C) 2000 SCI Group

KEYWORDS

DESCRIPTION
   Long description...

WARNING

****************************************/

      class SimpleMaterial : public Material {
      public:
	 SimpleMaterial();
	 
	 ~SimpleMaterial();
 
      private:
      };

} // End namespace Uintah

#endif // __MPM_MATERIAL_H__
