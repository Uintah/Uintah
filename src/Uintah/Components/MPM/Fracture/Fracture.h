#ifndef __FRACTURE_H__
#define __FRACTURE_H__

#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/CCVariable.h>

#include <Uintah/Interface/DataWarehouseP.h>

namespace Uintah {

   class VarLabel;
   class ProcessorContext;

namespace MPM {

class Fracture {
public:
  enum CellStatus { HAS_ONE_BOUNDARY_SURFACE,
                    HAS_SEVERAL_BOUNDARY_SURFACE,
                    INTERIOR_CELL
                  };

  enum ParticleStatus { BOUNDARY_PARTICLE,
                        INTERIOR_PARTICLE
                      };

  void   materialDefectsInitialize(const Patch* patch,
                                   DataWarehouseP& new_dw);
  
  void   initializeFracture(const Patch* patch,
                           DataWarehouseP& new_dw);
  
  void   updateSurfaceNormalOfBoundaryParticle(
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw);
  
  void   labelSelfContactNodesAndCells (
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw);

  void   updateParticleInformationInContactCells (
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw);

  void   updateNodeInformationInContactCells (
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw);

  void   crackGrow(
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw);

         Fracture();
	 Fracture(ProblemSpecP& ps, SimulationStateP& d_sS);
                
private:
  void   labelCellSurfaceNormal (
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw);

  void   labelSelfContactNodes(
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw);

  static bool isSelfContactNode(const IntVector& nodeIndex,const Patch* patch,
    const CCVariable<Vector>& cSurfaceNormal);

  static Fracture::CellStatus  cellStatus(
           const Vector& cellSurfaceNormal);
  static void setCellStatus(Fracture::CellStatus status,
           Vector* cellSurfaceNormal);

  static Fracture::ParticleStatus  particleStatus(
           const Vector& particleSurfaceNormal);
  static void setParticleStatus(Fracture::ParticleStatus status,
           Vector* particleSurfaceNormal);

  double           d_averageMicrocrackLength;
  double           d_materialToughness;
  SimulationStateP d_sharedState;
};

} //namespace MPM
} //namespace Uintah

#endif //__FRACTURE_H__

// $Log$
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
