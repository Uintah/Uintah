#ifndef __FRACTURE_H__
#define __FRACTURE_H__

#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include "Lattice.h"
#include "Cell.h"
#include "LeastSquare.h"
#include "CubicSpline.h"

#include <Packages/Uintah/Core/Math/Matrix3.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>

namespace Uintah {
   class VarLabel;
   class ProcessorGroup;

class Fracture {
public:

  virtual void   initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouse* new_dw) = 0;

  virtual void   computeConnectivity(
                  const PatchSubset* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw) = 0;

  virtual void   computeCrackExtension(
                  const PatchSubset* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw) = 0;

  virtual void   computeFracture(
                  const PatchSubset* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw) = 0;

  virtual void   computeBoundaryContact(
                  const PatchSubset* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw) = 0;

  int            getConstraint() const;

  Fracture(ProblemSpecP& ps);
  virtual ~Fracture();

protected:
  MPMLabel*        lb;
  int              d_constraint;
};

} // End namespace Uintah

#endif //__FRACTURE_H__

