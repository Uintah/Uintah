#ifndef __MPM_MATERIAL_H__
#define __MPM_MATERIAL_H__

// Do not EVER put a #include for anything in CCA/Components in here.
// Ask steve for a better way

#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

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
 class MPMLabel;
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

   // Default Constructor
   MPMMaterial();

   // Standard MPM Material Constructor
   MPMMaterial(ProblemSpecP&, MPMLabel* lb, MPMFlags* flags,SimulationStateP&);
	 
   ~MPMMaterial();

   /*!  Create a copy of the material without the associated geometry */
   void copyWithoutGeom(const MPMMaterial* mat, MPMFlags* flags,
                        SimulationStateP& sharedState);
	 
   //////////
   // Return correct constitutive model pointer for this material
   ConstitutiveModel* getConstitutiveModel() const;

   // Return correct burn model pointer for this material
   particleIndex countParticles(const Patch* patch);

   void createParticles(particleIndex numParticles,
			CCVariable<short int>& cellNAPID,
			const Patch*,
			DataWarehouse* new_dw);


   ParticleCreator* getParticleCreator();

   double getInitialDensity() const;

   // Get the specific heats at room temperature
   double getInitialCp() const;
   double getInitialCv() const;

   // for temperature dependent plasticity models
   double getRoomTemperature() const;
   double getMeltTemperature() const;

   bool getIsRigid() const;

   int nullGeomObject() const;

   // For MPMICE
   double getGamma() const;
   void initializeCCVariables(CCVariable<double>& rhom,
			      CCVariable<double>& rhC,
			      CCVariable<double>& temp,   
			      CCVariable<Vector>& vCC,
			      int numMatls,
			      const Patch* patch);

   void initializeDummyCCVariables(CCVariable<double>& rhom,
			           CCVariable<double>& rhC,
			           CCVariable<double>& temp,   
			           CCVariable<Vector>& vCC,
			           int numMatls,
			           const Patch* patch);

 private:

   MPMLabel* lb;
   // Specific constitutive model associated with this material
   ConstitutiveModel* d_cm;

   ParticleCreator* d_particle_creator;

   double d_density;

   // Specific heats at constant pressure and constant volume
   // (values at room temperature - [273.15 + 20] K)
   double d_Cp, d_Cv;

   // for temperature dependent plasticity models
   double d_troom;
   double d_tmelt;

   // for implicit rigid body contact
   bool d_is_rigid;
   bool d_includeFlowWork;

   std::vector<GeometryObject*> d_geom_objs;

   // Prevent copying of this class
   // copy constructor
   MPMMaterial(const MPMMaterial &mpmm);
   MPMMaterial& operator=(const MPMMaterial &mpmm);

   ///////////////////////////////////////////////////////////////////////////
   //
   // The standard set of initialization actions except particlecreator
   //
   void standardInitialization(ProblemSpecP& ps, MPMLabel* lb, 
                               MPMFlags* flags,SimulationStateP& sharedState);
 };

} // End namespace Uintah

#endif // __MPM_MATERIAL_H__
