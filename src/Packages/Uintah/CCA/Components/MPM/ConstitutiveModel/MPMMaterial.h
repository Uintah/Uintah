#ifndef __MPM_MATERIAL_H__
#define __MPM_MATERIAL_H__

#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/GeometryPiece.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <vector>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace Uintah {

using namespace SCIRun;

 class Patch;
 class DataWarehouse;
 class VarLabel;
 class GeometryObject;
 class GeometryPiece;
 class ConstitutiveModel;
 class Burn;
 class EquationOfState;
 class ParticleCreator;
      
/**************************************
     
CLASS
   MPMMaterial

   Short description...

GENERAL INFORMATION

   MPMMaterial.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   Copyright (C) 2000 SCI Group

KEYWORDS
   MPM_Material

DESCRIPTION
   Long description...

WARNING

****************************************/

      class MPMMaterial : public Material {
      public:
	 MPMMaterial(ProblemSpecP&, MPMLabel* lb, int n8or27,string integrat);
	 
	 ~MPMMaterial();
	 
	 //////////
	 // Return correct constitutive model pointer for this material
	 ConstitutiveModel* getConstitutiveModel() const;

	 // Return correct burn model pointer for this material
	 Burn* getBurnModel();

	 // Return correct EOS model pointer for this material
	 EquationOfState* getEOSModel() const;
	 
	 particleIndex countParticles(const Patch* patch);

	 void createParticles(particleIndex numParticles,
			      CCVariable<short int>& cellNAPID,
			      const Patch*,
			      DataWarehouse* new_dw);

         //for HeatConductionModel
         double getThermalConductivity() const;
         double getSpecificHeat() const;
         double getHeatTransferCoefficient() const;
         double getInitialDensity() const;

         // for temperature dependent plasticity models
         double getRoomTemperature() const;
         double getMeltTemperature() const;

         bool getIsRigid() const;

         // For MPMICE
         double getGamma() const;
         void initializeCCVariables(CCVariable<double>& rhom,
                                    CCVariable<double>& rhC,
                                    CCVariable<double>& temp,   
                                    CCVariable<Vector>& vCC,
                                    int numMatls,
                                    const Patch* patch);
      private:

	 MPMLabel* lb;
	 // Specific constitutive model associated with this material
	 ConstitutiveModel* d_cm;

         // Burn model
	 Burn* d_burn;

	 // EOS model
	 EquationOfState* d_eos;


	 ParticleCreator* d_particle_creator;

	 double d_density;

         //for HeatConductionModel
         double d_thermalConductivity;
         double d_specificHeat;
         
         double d_gamma;

         // for temperature dependent plasticity models
         double d_troom;
         double d_tmelt;

         // for implicit rigid body contact
         bool d_is_rigid;

         //for ThermalContactModel
         double d_heatTransferCoefficient;

	 std::vector<GeometryObject*> d_geom_objs;

	 // Prevent copying of this class
	 // copy constructor
	 MPMMaterial(const MPMMaterial &mpmm);
	 MPMMaterial& operator=(const MPMMaterial &mpmm);
      };

} // End namespace Uintah

#endif // __MPM_MATERIAL_H__
