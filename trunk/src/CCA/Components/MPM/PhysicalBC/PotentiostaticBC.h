/*
 * PotentiostaticBC.h
 *
 *  Created on: Oct 4, 2017
 *      Author: jbhooper
 */

#ifndef SRC_CCA_COMPONENTS_MPM_PHYSICALBC_POTENTIOSTATICBC_H_
#define SRC_CCA_COMPONENTS_MPM_PHYSICALBC_POTENTIOSTATICBC_H_

#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <CCA/Components/MPM/PhysicalBC/LoadCurve.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <Core/Geometry/Vector.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Grid.h>
#include <Core/Math/Matrix3.h>

namespace Uintah {

class GeometryPiece;
class ParticleCreator;

/*

CLASS
  PotentiostaticBC

    Uniform external mass flux boundary condition for MPM.

GENERAL INFORMATION
  PotentiostaticBC.h

  Justin Hooper
  Department of Materials Science and Engineering, University of Utah

DESCRIPTION

   Allows for imposition of a potentiostatic (constant mass flux) boundary
   condition which is applied to particles on the surface particles present
   on the surface of the geometry represented by the intersection of the material
   points within the system and the volume of the boundary.  A buffer of
   length(Dot (0.5*dX,0.5*dX)) is used to buffer the boundary of the boundary
   condition externally by default, but this can be set to an arbitrary positive
   value if desired.

WARNING
   The particles marked as surface particles are determined at initiation of the
   problem execution.  As such there is no guarantee they will remain at a
   surface throughout the entire problem run length.  The assumption is therefore
   made that the surface remains topologically consistent and exposed throughout
   the entire time domain.

 */

  class PotentiostaticBC: public MPMPhysicalBC {
    public:
                          PotentiostaticBC(
                                                                  ProblemSpecP  & ps
                                                          , const GridP                 & grid
                                                          , const MPMFlags              * flag);
                         ~PotentiostaticBC();

          // Required functions from MPMPhysicalBC.h
          virtual std::string   getType() const;
          virtual void                  outputProblemSpec(ProblemSpecP & ps);

          // Locate and flag the material points to which this boundary condition
          //   will be applied (i.e. the surface points).
                          bool                  flagMaterialPoint(
                                                                                           const Point  &       p
                                                                                         , const Vector &   buffer = Vector(1.0,1.0,1.0));

         //  Get the load curve number for this BC.
         inline   int                   loadCurveID() const { return d_loadCurve->getID(); }

         // Get the surface geometry descriptor
         inline GeometryPiece* getSurface() const { return d_surface; }

         //  Get the surface type
         inline std::string getSurfaceType() const { return d_surfaceType; }

         // Set the number of material points on the surface
         inline void numMaterialPoints(long numPoints) { d_numMaterialPoints = numPoints ; }

         // Query the number of material points on the surface.
         inline long numMaterialPoints() { return d_numMaterialPoints; }

         double particleFlux(double time, double area) const;

    private:

         // Loading points for material (time, flux magnitude pairs).
          LoadCurve<double>* d_loadCurve;

          GeometryPiece*     d_surface;
          std::string        d_surfaceType;
          long               d_numMaterialPoints;
  };


}



#endif /* SRC_CCA_COMPONENTS_MPM_PHYSICALBC_POTENTIOSTATICBC_H_ */
