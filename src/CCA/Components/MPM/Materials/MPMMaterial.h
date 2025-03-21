/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __MPM_MATERIAL_H__
#define __MPM_MATERIAL_H__

// Do not EVER put a #include for anything in another CCA/Components in here.
// (#includes of other MPM files is ok.  However, if you #include'd ARCHES
// or something, then a circular dependency would be created.)

#include <CCA/Components/MPM/Core/MPMFlags.h>

#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <vector>

namespace Uintah {

 class Patch;
 class DataWarehouse;
 class VarLabel;
 class GeometryObject;
 class ConstitutiveModel;
 class DamageModel;
 class ErosionModel;
 class MPMLabel;
 class ParticleCreator;
 class ScalarDiffusionModel;

      
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
   MPMMaterial(ProblemSpecP&, MaterialManagerP& ss, MPMFlags* flags,
               const bool isRestart);
         
   ~MPMMaterial();

   virtual void registerParticleState( std::vector<std::vector<const VarLabel* > > &PState,
                                       std::vector<std::vector<const VarLabel* > > &PState_preReloc );

   virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

   /*!  Create a copy of the material without the associated geometry */
   void copyWithoutGeom(ProblemSpecP& ps,const MPMMaterial* mat,
                        MPMFlags* flags);
         
   //////////
   // Return correct constitutive model pointer for this material
   ConstitutiveModel* getConstitutiveModel() const;
   
   void deleteGeomObjects( );
   
   void set_pLocalizedComputed( const bool ans);
  
   bool is_pLocalizedPreComputed() const;
   
   DamageModel* getDamageModel() const;
   
   ErosionModel* getErosionModel() const;

   ScalarDiffusionModel* getScalarDiffusionModel() const;


   particleIndex createParticles(
                        CCVariable<int>& cellNAPID,
                        const Patch*,
                        DataWarehouse* new_dw);


   ParticleCreator* getParticleCreator();

   double getInitialDensity() const;

   // Get the specific heats at room temperature
   double getInitialCp() const;
   double getInitialCv() const;
   double getThermalExpansionCoefficient() const;

   // for temperature dependent plasticity models
   double getRoomTemperature() const;
   double getMeltTemperature() const;

   bool getIsRigid() const;
   void setIsRigid(const bool is_rigid);

   bool getIsFTM() const;
   void setIsFTM(const bool is_ftm);

   bool getIsActive() const;
   void setIsActive(const bool is_active);
   double getActivationTime() const;

   bool getPossibleAlphaMaterial() const;
   void setPossibleAlphaMaterial(const bool possible_alpha);

   double getSpecificHeat() const;
   double getThermalConductivity() const;

   int nullGeomObject() const;

   // MPM Hydro-mechanical coupling
   double getWaterDensity() const;
   double getPorosity() const;
   double getPermeability() const;
   double getInitialPorepressure() const;

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

   bool doConcReduction(){ return d_do_conc_reduction; };

 private:

   MPMLabel* d_lb;
   ConstitutiveModel*     d_cm;
   DamageModel*           d_damageModel;
   ErosionModel*          d_erosionModel;
   ScalarDiffusionModel*  d_sdm;
   ParticleCreator*       d_particle_creator;

   double d_density;
   double d_specificHeat;
   double d_thermalConductivity;
   bool   d_pLocalizedComputed  =  false;        // set to true if any task computes pLocalizedMPM or pLocalizedMPM_preReloc
   bool   d_allTriGeometry = false;
   bool   d_allFileGeometry = false;

   // Specific heats at constant pressure and constant volume
   // (values at room temperature - [273.15 + 20] K)
   double d_Cp, d_Cv;

   // for temperature dependent plasticity models
   double d_troom;
   double d_tmelt;

   // for thermal expansion (linear thermal expansion coefficient)
   double d_thermalExpCoeff;

   // for rigid body contact
   bool d_is_rigid;

   // for frictionLR contact model
   bool d_possible_alpha;

   // for undeformable force transmission
   bool d_is_force_transmitting_material;

   // for insert particles
   bool d_is_active;
   double d_activation_time;

   // Parameters related to MPM Hydro-mechanical coupling
   double d_waterdensity, d_porosity, d_permeability, d_initial_porepressure;

   // for autocycleflux boundary condtions
   bool d_do_conc_reduction;

   std::vector<GeometryObject*> d_geom_objs;

   // Prevent copying of this class
   // copy constructor
   MPMMaterial(const MPMMaterial &mpmm);
   MPMMaterial& operator=(const MPMMaterial &mpmm);

   ///////////////////////////////////////////////////////////////////////////
   //
   // The standard set of initialization actions except particlecreator
   //
   void standardInitialization(ProblemSpecP& ps, MaterialManagerP& ss,
                               MPMFlags* flags, const bool isRestart);
 };

} // End namespace Uintah

#endif // __MPM_MATERIAL_H__
