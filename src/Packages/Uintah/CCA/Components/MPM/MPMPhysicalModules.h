#ifndef UINTAH_MPM_MPMPHYSICALMODULES
#define UINTAH_MPM_MPMPHYSICALMODULES

#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>

namespace Uintah {
  class Contact;
  class ThermalContact;

/**************************************

CLASS
   MPMPhysicalModules
   
   Short description...

GENERAL INFORMATION

   MPMPhysicalModules.h

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   MPMPhysicalModules

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class MPMPhysicalModules {
   public:

      //Physical Models:
     // I don't like this - Steve
     // They are basically glorified common blocks/global variables
      static Contact*         contactModel;
      static ThermalContact*  thermalContactModel;

      static void build(const ProblemSpecP& prob_spec,SimulationStateP& sharedState);
     static void kill();
   };

} // End namespace Uintah

#endif //UINTAH_MPM_MPMPHYSICALMODULES
