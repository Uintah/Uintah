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
class Fracture;
class EquationOfState;
      
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
	 MPMMaterial(ProblemSpecP&, MPMLabel* lb);
	 
	 ~MPMMaterial();
	 
	 //////////
	 // Return correct constitutive model pointer for this material
	 ConstitutiveModel * getConstitutiveModel() const;

	 // Return correct burn model pointer for this material
	 Burn * getBurnModel();

	 // Return correct burn fracture pointer for this material
	 Fracture * getFractureModel() const;

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
				       ParticleVariable<double>& tensilestrengt,
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
				       ParticleVariable<double>& tensilestrengt,
				       ParticleVariable<long64>& particleID,
				       CCVariable<short int>& cellNAPID,
				       const Patch*,
				       ParticleVariable<Vector>& ptang1,
				       ParticleVariable<Vector>& ptang2,
				       ParticleVariable<Vector>& pnorm);

	 int checkForSurface(const GeometryPiece* piece,
				const Point p, const Vector dxpp);


         //for HeatConductionModel
         double getThermalConductivity() const;
         double getSpecificHeat() const;
         double getHeatTransferCoefficient() const;
         double getInitialDensity() const;

         //for FractureModel
	 double getPressureRate() const;
	 double getExplosivePressure() const;
	 double getInitialPressure() const;

         // For MPMICE
         double getGamma() const;
				
      private:

	 // Specific constitutive model associated with this material
	 ConstitutiveModel *d_cm;

         bool d_membrane;

         // Burn model
	 Burn *d_burn;

         // Fracture model
	 Fracture *d_fracture;

	 // EOS model
	 EquationOfState* d_eos;
	 
	 double d_density;

         //for HeatConductionModel
         double d_thermalConductivity;
         double d_specificHeat;
         
         double d_gamma;

         //for ThermalContactModel
         double d_heatTransferCoefficient;

         //for FractureModel
         double d_pressureRate;
	 double d_explosivePressure;
	 double d_initialPressure;

	 std::vector<GeometryObject*> d_geom_objs;

	 MPMLabel* lb;

	 // Prevent copying of this class
	 // copy constructor
	 MPMMaterial(const MPMMaterial &mpmm);
	 MPMMaterial& operator=(const MPMMaterial &mpmm);
      };

} // End namespace Uintah

#endif // __MPM_MATERIAL_H__
