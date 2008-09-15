#ifndef __ANGIO_MATERIAL_H__
#define __ANGIO_MATERIAL_H__

// Do not EVER put a #include for anything in CCA/Components in here.
// Ask steve for a better way

#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/CCA/Components/MPM/uintahshare.h>
namespace Uintah {

using namespace SCIRun;

 class Patch;
 class DataWarehouse;
 class VarLabel;
 class GeometryObject;
 class AngioLabel;
 class AngioFlags;
 class AngioParticleCreator;

      
/**************************************
     
CLASS
   AngioMaterial

   Contains some basic information for the vessel material

GENERAL INFORMATION

   AngioMaterial.h

   James Guilkey
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   Copyright (C) 2000 SCI Group

KEYWORDS
   Angio Material

DESCRIPTION
   Long description...

WARNING

****************************************/

 class UINTAHSHARE AngioMaterial : public Material {
 public:

   // Default Constructor
   AngioMaterial();

   // Standard Angio Material Constructor
   AngioMaterial(ProblemSpecP&, SimulationStateP& ss, AngioFlags* flags);

   ~AngioMaterial();

   virtual void registerParticleState(SimulationState* ss);

   virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

   /*!  Create a copy of the material without the associated geometry */
   void copyWithoutGeom(ProblemSpecP& ps,const AngioMaterial* mat,
                        AngioFlags* flags);
         
   // Return correct burn model pointer for this material
   particleIndex countParticles(const Patch* patch);

   void createParticles(particleIndex numParticles,
                        CCVariable<short int>& cellNAPID,
                        const Patch*,
                        DataWarehouse* new_dw);

   AngioParticleCreator* getParticleCreator();

   double getInitialDensity() const;

   int nullGeomObject() const;

 private:

   AngioLabel* d_lb;
   AngioParticleCreator* d_particle_creator;

   double d_density;
   std::string d_init_frag_file;

   std::vector<GeometryObject*> d_geom_objs;

   // Prevent copying of this class
   // copy constructor
   AngioMaterial(const AngioMaterial &mpmm);
   AngioMaterial& operator=(const AngioMaterial &mpmm);

   ///////////////////////////////////////////////////////////////////////////
   //
   // The standard set of initialization actions except particlecreator
   //
   void standardInitialization(ProblemSpecP& ps, AngioFlags* flags);
 };

} // End namespace Uintah

#endif // __ANGIO_MATERIAL_H__
