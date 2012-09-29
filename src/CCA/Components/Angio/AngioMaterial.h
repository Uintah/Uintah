/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __ANGIO_MATERIAL_H__
#define __ANGIO_MATERIAL_H__

// Do not EVER put a #include for anything in CCA/Components in here.
// Ask steve for a better way

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

   
KEYWORDS
   Angio Material

DESCRIPTION
   Long description...

WARNING

****************************************/

 class AngioMaterial : public Material {
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
