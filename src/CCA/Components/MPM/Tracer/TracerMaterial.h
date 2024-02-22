/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#ifndef __TRACER_MATERIAL_H__
#define __TRACER_MATERIAL_H__

// Do not EVER put a #include for anything in CCA/Components in here.
// Ask steve for a better way

#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Components/MPM/Tracer/Tracer.h>
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
 class MPMLabel;
 class MPMFlags;

      
/**************************************
     
CLASS
   TracerMaterial

   Short description...

GENERAL INFORMATION

   TracerMaterial.h

   Jim Guilkey
   Laird Avenue Consulting   

KEYWORDS
   Tracer_Material

DESCRIPTION
   Long description...

WARNING

****************************************/

 class TracerMaterial : public Material {
 public:

   // Default Constructor
   TracerMaterial();

   // Standard Tracer Material Constructor
   TracerMaterial(ProblemSpecP&, MaterialManagerP& ss, MPMFlags* flags);

   ~TracerMaterial();

   virtual void registerParticleState(
             std::vector<std::vector<const VarLabel* > > &TracerState,
             std::vector<std::vector<const VarLabel* > > &TracerState_preReloc);

   virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

   /*!  Create a copy of the material without the associated geometry */
   void copyWithoutGeom(ProblemSpecP& ps,const TracerMaterial* mat,
                        MPMFlags* flags);
         
   // Access functions
   Tracer* getTracer();
   std::string getTracerFilename() const;

   int getAssociatedMaterial() const;

 private:

   MPMLabel* d_lb;
   Tracer* d_tracer;

   std::string d_tracer_filename;
   int d_associated_material;

   // Prevent copying of this class
   // copy constructor
   TracerMaterial(const TracerMaterial &tm);
   TracerMaterial& operator=(const TracerMaterial &tm);

   ///////////////////////////////////////////////////////////////////////////
   // The standard set of initialization actions except particlecreator
   //
   void standardInitialization(ProblemSpecP& ps, MPMFlags* flags);
 };

} // End namespace Uintah

#endif // __TRACER_MATERIAL_H__
