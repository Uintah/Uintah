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

#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>

namespace Uintah {
   class VarLabel;
   class ProcessorGroup;

class Fracture {
public:

  virtual void   initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouseP& new_dw) = 0;

  virtual void   computeConnectivity(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw) = 0;

  virtual void   computeFracture(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw) = 0;

  virtual void   computeBoundaryContact(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouseP& old_dw, 
		  DataWarehouseP& new_dw) = 0;

  Fracture(ProblemSpecP& ps);
  virtual ~Fracture();

protected:
  MPMLabel*        lb;
};

} // End namespace Uintah

#endif //__FRACTURE_H__

