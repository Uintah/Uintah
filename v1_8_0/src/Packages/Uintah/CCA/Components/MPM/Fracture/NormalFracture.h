#ifndef _MPM_NormalFracture
#define _MPM_NormalFracture

#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include "Fracture.h"
#include "Lattice.h"
#include "Cell.h"
#include "LeastSquare.h"
#include "CubicSpline.h"
#include "BoundaryBand.h"

#include <Packages/Uintah/Core/Math/Matrix3.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>

namespace Uintah {
   class VarLabel;
   class ProcessorGroup;

class NormalFracture : public Fracture {
public:

  void   initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouse* new_dw);

  void   computeBoundaryContact(
                  const PatchSubset* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw);

  void   computeConnectivity(
                  const PatchSubset* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw);

  void   computeCrackExtension(
                  const PatchSubset* patches,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw);

  void   computeFracture(
                  const PatchSubset* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw);

         NormalFracture(ProblemSpecP& ps);
	~NormalFracture();

  static double connectionRadius(double volume);

  //for debugging
  static bool   isDebugParticle(const Point& p);

private:
  void BoundaryBandConnection(
       const vector<BoundaryBand>& pBoundaryBand_pg,
       particleIndex pIdx_pg,
       const Point& part,
       const Point& node,
       int& conn) const;

  void VisibilityConnection(
       const ParticlesNeighbor& particles,
       particleIndex pIdx_pg,
       const Point& node,
       const ParticleVariable<int>& pCrackEffective_pg,
       const ParticleVariable<Vector>& pCrackNormal_pg,
       const ParticleVariable<double>& pVolume_pg,
       const ParticleVariable<Point>& pX_pg,
       int& conn) const;

  void ContactConnection(
       const ParticlesNeighbor& particles,
       particleIndex pIdx_pg,
       const Point& node,
       const ParticleVariable<Vector>& pTouchNormal_pg,
       const ParticleVariable<double>& pVolume_pg,
       const ParticleVariable<Point>& pX_pg,
       int& conn,
       Vector& pContactNormal) const;

};

} // End namespace Uintah

#endif

