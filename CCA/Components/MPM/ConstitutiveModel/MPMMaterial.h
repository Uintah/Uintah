#ifndef __MPM_MATERIAL_H__
#define __MPM_MATERIAL_H__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/GeometryPiece.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

#include <vector>

#define IMPLICIT
//#undef IMPLICIT

namespace SCIRun {
  class Point;
  class Vector;
}

namespace Uintah {

using namespace SCIRun;

 class Patch;
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
	 ConstitutiveModel * getConstitutiveModel() const;

	 // Return correct burn model pointer for this material
	 Burn * getBurnModel();

	 // Return correct EOS model pointer for this material
	 EquationOfState* getEOSModel() const;
	 
	 particleIndex countParticles(const Patch*) const;
	 particleIndex countParticles(GeometryObject* obj,
				      const Patch*) const;
	 void createParticles(particleIndex numParticles,
			      CCVariable<short int>& cellNAPID,
			      const Patch*,
			      DataWarehouse* new_dw);

	 particleIndex createParticles(GeometryObject* obj,
				       particleIndex start,
				       ParticleVariable<Point>& position,
				       ParticleVariable<Vector>& velocity,
				       ParticleVariable<Vector>& pexternalforce,
				       ParticleVariable<double>& mass,
				       ParticleVariable<double>& volume,
				       ParticleVariable<double>& temperature,
				       ParticleVariable<Vector>& size,
#ifdef IMPLICIT
				       ParticleVariable<Vector>& pacceleration,
				       ParticleVariable<double>& pvolumeold,
				       ParticleVariable<Matrix3>& bElBar,
#endif
				       ParticleVariable<long64>& particleID,
				       CCVariable<short int>& cellNAPID,
				       const Patch*);

	 particleIndex createParticles(GeometryObject* obj,
				       particleIndex start,
				       ParticleVariable<Point>& position,
				       ParticleVariable<Vector>& velocity,
				       ParticleVariable<Vector>& pexternalforce,
				       ParticleVariable<double>& mass,
				       ParticleVariable<double>& volume,
				       ParticleVariable<double>& temperature,
				       ParticleVariable<Vector>& size,
				       ParticleVariable<long64>& particleID,
				       CCVariable<short int>& cellNAPID,
				       const Patch*,
				       ParticleVariable<Vector>& ptang1,
				       ParticleVariable<Vector>& ptang2,
				       ParticleVariable<Vector>& pnorm);

         //for HeatConductionModel
         double getThermalConductivity() const;
         double getSpecificHeat() const;
         double getHeatTransferCoefficient() const;
         double getInitialDensity() const;

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
	 ConstitutiveModel *d_cm;

         // Burn model
	 Burn *d_burn;

	 // EOS model
	 EquationOfState* d_eos;

         bool d_membrane;
#if 0
	 ParticleCreator* d_particle_creator;
#endif 
	 double d_density;

         //for HeatConductionModel
         double d_thermalConductivity;
         double d_specificHeat;
         
         double d_gamma;

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
