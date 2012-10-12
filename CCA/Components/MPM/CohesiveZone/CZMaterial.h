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

#ifndef __CZ_MATERIAL_H__
#define __CZ_MATERIAL_H__

// Do not EVER put a #include for anything in CCA/Components in here.
// Ask steve for a better way

#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Components/MPM/CohesiveZone/CohesiveZone.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
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
 class MPMLabel;
 class MPMFlags;

      
/**************************************
     
CLASS
   CZMaterial

   Short description...

GENERAL INFORMATION

   CZMaterial.h

   Jim Guilkey
   Perforating Research
   Schlumberger   

KEYWORDS
   CZ_Material

DESCRIPTION
   Long description...

WARNING

****************************************/

 class CZMaterial : public Material {
 public:

   // Default Constructor
   CZMaterial();

   // Standard CZ Material Constructor
   CZMaterial(ProblemSpecP&, SimulationStateP& ss, MPMFlags* flags);
         
   ~CZMaterial();

   virtual void registerParticleState(SimulationState* ss);

   virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

   /*!  Create a copy of the material without the associated geometry */
   void copyWithoutGeom(ProblemSpecP& ps,const CZMaterial* mat,
                        MPMFlags* flags);
         
   // Access functions
   CohesiveZone* getCohesiveZone();
   double getCharLengthNormal() const;
   double getCharLengthTangential() const;
   double getCohesiveNormalStrength() const;
   double getCohesiveTangentialStrength() const;
   string getCohesiveFilename() const;
   bool getDoRotation() const;

   void computeRotationMatrix(Matrix3& Rotation, Matrix3& Rotation_tang,
                              const Vector& norm, const Vector czsep) const;

 private:

   MPMLabel* d_lb;
   CohesiveZone* d_cohesive_zone;

   double d_delta_n;
   double d_delta_t;
   double d_sig_max;
   double d_tau_max;
   bool d_do_rotation;
   string d_cz_filename;

   // Prevent copying of this class
   // copy constructor
   CZMaterial(const CZMaterial &czm);
   CZMaterial& operator=(const CZMaterial &czm);

   ///////////////////////////////////////////////////////////////////////////
   // The standard set of initialization actions except particlecreator
   //
   void standardInitialization(ProblemSpecP& ps, MPMFlags* flags);
 };

} // End namespace Uintah

#endif // __CZ_MATERIAL_H__
