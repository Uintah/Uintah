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

#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>

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

  void   computeFracture(
                  const PatchSubset* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw);

         NormalFracture(ProblemSpecP& ps);
	~NormalFracture();

private:
};

} // End namespace Uintah

#endif

