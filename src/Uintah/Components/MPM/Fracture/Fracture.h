#ifndef __FRACTURE_H__
#define __FRACTURE_H__

#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include "Lattice.h"
#include "Cell.h"
#include "LeastSquare.h"
#include "CubicSpline.h"

#include <Uintah/Components/MPM/Util/Matrix3.h>

#include <Uintah/Interface/DataWarehouseP.h>

namespace Uintah {

   class VarLabel;
   class ProcessorGroup;

namespace MPM {

class Fracture {
public:

  void   initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouseP& new_dw);

  void   computerNodesVisibility(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw);

  void   crackGrow(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw);

	 Fracture(ProblemSpecP& ps);
        ~Fracture();
                
private:
  double           d_tensileStrength;

  MPMLabel*        lb;
};

} //namespace MPM
} //namespace Uintah

#endif //__FRACTURE_H__

// $Log$
// Revision 1.28  2000/09/12 16:52:11  tan
// Reorganized crack surface contact force algorithm.
//
// Revision 1.27  2000/09/11 18:55:51  tan
// Crack surface contact force is now considered in the simulation.
//
// Revision 1.26  2000/09/09 19:34:16  tan
// Added MPMLabel::pVisibilityLabel and SerialMPM::computerNodesVisibility().
//
// Revision 1.25  2000/09/05 19:38:10  tan
// Fracture starts to run in Uintah/MPM!
//
// Revision 1.24  2000/09/05 05:13:09  tan
// Moved Fracture Model to MPMMaterial class.
//
// Revision 1.23  2000/08/09 03:18:02  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.22  2000/07/06 16:58:45  tan
// Least square interpolation added for particle velocities and stresses
// updating.
//
// Revision 1.21  2000/07/05 23:43:37  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.20  2000/07/05 21:37:43  tan
// Filled in the function of updateParticleInformationInContactCells.
//
// Revision 1.19  2000/06/23 16:49:23  tan
// Added LeastSquare Approximation and Lattice for neighboring algorithm.
//
// Revision 1.18  2000/06/23 01:37:59  tan
// Moved material property toughness to Fracture class.
//
// Revision 1.17  2000/06/17 07:06:40  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.16  2000/06/05 02:07:39  tan
// Finished labelSelfContactNodesAndCells(...).
//
// Revision 1.15  2000/06/04 23:55:29  tan
// Added labelSelfContactCells(...) to label the self-contact cells
// according to the nodes self-contact information.
//
// Revision 1.14  2000/06/02 21:54:12  tan
// Finished function labelSelfContactNodes(...) to label the gSalfContact
// according to the cSurfaceNormal information.
//
// Revision 1.13  2000/06/02 21:12:07  tan
// Added function isSelfContactNode(...) to determine if a node is a
// self-contact node.
//
// Revision 1.12  2000/06/02 00:12:58  tan
// Added ParticleStatus to determine if a particle is a BOUNDARY_PARTICLE
// or a INTERIOR_PARTICLE.
//
// Revision 1.11  2000/06/01 23:55:47  tan
// Added CellStatus to determine if a cell HAS_ONE_BOUNDARY_SURFACE,
// HAS_SEVERAL_BOUNDARY_SURFACE or is INTERIOR cell.
//
// Revision 1.10  2000/05/30 20:19:13  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.9  2000/05/30 04:36:46  tan
// Using MPMLabel instead of VarLabel.
//
// Revision 1.8  2000/05/15 18:58:53  tan
// Initialized NCVariables and CCVaribles for Fracture.
//
// Revision 1.7  2000/05/12 01:46:21  tan
// Added initializeFracture linked to SerialMPM's actuallyInitailize.
//
// Revision 1.6  2000/05/11 20:10:18  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.5  2000/05/10 18:32:22  tan
// Added member funtion to label self-contact cells.
//
// Revision 1.4  2000/05/10 05:04:39  tan
// Basic structure of fracture class.
//
