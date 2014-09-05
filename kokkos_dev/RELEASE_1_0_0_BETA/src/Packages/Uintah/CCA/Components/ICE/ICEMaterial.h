#ifndef __ICE_MATERIAL_H__
#define __ICE_MATERIAL_H__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfState.h>
#include <Packages/Uintah/CCA/Components/ICE/GeometryObject2.h>

#include <vector>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace Uintah {
using namespace SCIRun;
      
/**************************************
     
CLASS
   ICEMaterial

   Short description...

GENERAL INFORMATION

   ICEMaterial.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   Copyright (C) 2000 SCI Group

KEYWORDS
   ICE

DESCRIPTION
   Long description...

WARNING

****************************************/

      class ICEMaterial : public Material {
      public:
	 ICEMaterial(ProblemSpecP&);
	 
	 ~ICEMaterial();
	 
	 //////////
	 // Return correct EOS model pointer for this material
	 EquationOfState* getEOS() const;

         //for HeatConductionModel
        double getThermalConductivity() const;
        double getSpecificHeat() const;
        double getHeatTransferCoefficient() const;
        double getGamma() const;
	 double getViscosity() const;
	 double getSpeedOfSound() const;

	 void initializeCells(CCVariable<double>& rhom,
                           CCVariable<double>&sp_vol,
                           CCVariable<double>& rhC,
			      CCVariable<double>& temp, 
			      CCVariable<double>& ss,    CCVariable<double>& vsc,
			      CCVariable<double>& volf,  CCVariable<Vector>& vCC,
                           CCVariable<double>& press,
                           int numMatls,
			      const Patch* patch, DataWarehouseP& new_dw);

      private:

	 // Specific constitutive model associated with this material
	 EquationOfState *d_eos;

        double d_thermalConductivity;
        double d_specificHeat;
	 double d_heatTransferCoefficient;
	 double d_speed_of_sound;
	 double d_viscosity;
	 double d_gamma;
         
	 std::vector<GeometryObject2*> d_geom_objs;

	 ICELabel* lb;

	 // Prevent copying of this class
	 // copy constructor
	 ICEMaterial(const ICEMaterial &icem);
	 ICEMaterial& operator=(const ICEMaterial &icem);        
        
      };
} // End namespace Uintah

#endif // __ICE_MATERIAL_H__
