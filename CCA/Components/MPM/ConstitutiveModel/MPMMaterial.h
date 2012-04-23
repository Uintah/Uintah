/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef __MPM_MATERIAL_H__
#define __MPM_MATERIAL_H__

// Do not EVER put a #include for anything in another CCA/Components in here.
// (#includes of other MPM files is ok.  However, if you #include'd ARCHES
// or something, then a circular dependency would be created.)

#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <vector>

namespace Uintah {

using namespace SCIRun;

 class Patch;
 class DataWarehouse;
 class VarLabel;
 class GeometryObject;
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
   MPMMaterial(ProblemSpecP&, SimulationStateP& ss, MPMFlags* flags);
         
   ~MPMMaterial();

   virtual void registerParticleState(SimulationState* ss);

   virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

   /*!  Create a copy of the material without the associated geometry */
   void copyWithoutGeom(ProblemSpecP& ps,const MPMMaterial* mat,
                        MPMFlags* flags);
         
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

   bool getIncludeFlowWork() const;
   double getSpecificHeat() const;
   double getThermalConductivity() const;

   int nullGeomObject() const;


   // For MPMICE
   double getGamma() const;
   void initializeCCVariables(CCVariable<double>& rhom,
                              CCVariable<double>& rhC,
                              CCVariable<double>& temp,   
                              CCVariable<Vector>& vCC,
                              CCVariable<double>& vfCC,
                              const Patch* patch);

   void initializeDummyCCVariables(CCVariable<double>& rhom,
                                   CCVariable<double>& rhC,
                                   CCVariable<double>& temp,   
                                   CCVariable<Vector>& vCC,
                                   CCVariable<double>& vfCC,
                                   const Patch* patch);

 private:

   MPMLabel* d_lb;
   ConstitutiveModel* d_cm;
   ParticleCreator* d_particle_creator;

   double d_density;
   bool d_includeFlowWork;
   double d_specificHeat;
   double d_thermalConductivity;

   // Specific heats at constant pressure and constant volume
   // (values at room temperature - [273.15 + 20] K)
   double d_Cp, d_Cv;

   // for temperature dependent plasticity models
   double d_troom;
   double d_tmelt;

   // for implicit rigid body contact
   bool d_is_rigid;

   std::vector<GeometryObject*> d_geom_objs;

   // Prevent copying of this class
   // copy constructor
   MPMMaterial(const MPMMaterial &mpmm);
   MPMMaterial& operator=(const MPMMaterial &mpmm);

   ///////////////////////////////////////////////////////////////////////////
   //
   // The standard set of initialization actions except particlecreator
   //
   void standardInitialization(ProblemSpecP& ps, MPMFlags* flags);
 };

} // End namespace Uintah

#endif // __MPM_MATERIAL_H__
