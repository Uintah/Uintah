#ifndef Uintah_MPM_MPMPhysicalModules
#define Uintah_MPM_MPMPhysicalModules

#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/ProblemSpecP.h>

namespace Uintah {

namespace MPM {

  class HeatConduction;
  class Fracture;
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
      static HeatConduction*  heatConductionModel;
      static Fracture*        fractureModel;
      static Contact*         contactModel;
      static ThermalContact*  thermalContactModel;

      static void build(const ProblemSpecP& prob_spec,SimulationStateP& sharedState);
   };

} //namespace MPM
   
} // end namespace Uintah

#endif //Uintah_MPM_MPMPhysicalModules

//
// $Log$
// Revision 1.1  2000/06/22 21:21:52  tan
// MPMPhysicalModules class is created to handle all the physical modules
// in MPM, currently those physical submodules include HeatConduction,
// Fracture, Contact, and ThermalContact.
//
