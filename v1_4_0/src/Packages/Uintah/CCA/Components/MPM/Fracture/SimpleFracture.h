#ifndef _MPM_SimpleFracture
#define _MPM_SimpleFracture

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

#include <Packages/Uintah/Core/Math/Matrix3.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>

namespace Uintah {
   class VarLabel;
   class ProcessorGroup;

class SimpleFracture : public Fracture {
public:

  void   initializeFractureModelData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouse* new_dw);

  void   computeNodeVisibility(
                  const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw);

  void   crackGrow(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw);

  void   stressRelease(const Patch* patch,
                  MPMMaterial* mpm_matl, 
		  DataWarehouse* old_dw, 
		  DataWarehouse* new_dw);

         SimpleFracture(ProblemSpecP& ps);
};

} // End namespace Uintah

#endif

