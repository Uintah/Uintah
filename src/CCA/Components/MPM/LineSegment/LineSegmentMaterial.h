/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

#ifndef __LINESEGMENT_MATERIAL_H__
#define __LINESEGMENT_MATERIAL_H__

// Do not EVER put a #include for anything in CCA/Components in here.
// Ask steve for a better way

#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Components/MPM/LineSegment/LineSegment.h>
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
   LineSegmentMaterial

   Short description...

GENERAL INFORMATION

   LineSegmentMaterial.h

   Jim Guilkey
   Laird Avenue Consulting   

KEYWORDS
   LineSegment_Material

DESCRIPTION
   Long description...

WARNING

****************************************/

 class LineSegmentMaterial : public Material {
 public:

   // Default Constructor
   LineSegmentMaterial();

   // Standard LineSegment Material Constructor
   LineSegmentMaterial(ProblemSpecP&, MaterialManagerP& ss, MPMFlags* flags);

   ~LineSegmentMaterial();

   virtual void registerParticleState(
             std::vector<std::vector<const VarLabel* > > &LineSegmentState,
             std::vector<std::vector<const VarLabel* > > &LineSegmentState_preReloc);

   virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

   /*!  Create a copy of the material without the associated geometry */
   void copyWithoutGeom(ProblemSpecP& ps,const LineSegmentMaterial* mat,
                        MPMFlags* flags);
         
   // Access functions
   LineSegment* getLineSegment();
   std::string getLineSegmentFilename() const;

   int getAssociatedMaterial() const;

 private:

   LineSegment* d_linesegment;

   std::string d_lineseg_filename;
   int d_associated_material;

   // Prevent copying of this class
   // copy constructor
   LineSegmentMaterial(const LineSegmentMaterial &tm);
   LineSegmentMaterial& operator=(const LineSegmentMaterial &tm);

   ///////////////////////////////////////////////////////////////////////////
   // The standard set of initialization actions except particlecreator
   //
   void standardInitialization(ProblemSpecP& ps, MPMFlags* flags);
 };

} // End namespace Uintah

#endif // __LINESEGMENT_MATERIAL_H__
