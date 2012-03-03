//----- Ray.cc ----------------------------------------------
#include <CCA/Components/Models/Radiation/RMCRT/MersenneTwister.h>
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/AMR_CoarsenRefine.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <time.h>

#include <sci_defs/cuda_defs.h>

using namespace Uintah;
using namespace std;

static DebugStream dbg("RAY",       false);
static DebugStream dbg2("RAY_DEBUG",false);
static DebugStream dbg_BC("RAY_BC", false);

//---------------------------------------------------------------------------
// Function: The GPU ray tracer kernel
//---------------------------------------------------------------------------
__global__ void rayTraceKernel ()
{
  // stub
}


//---------------------------------------------------------------------------
// Method: The GPU ray tracer
//---------------------------------------------------------------------------
void Ray::rayTraceGPU( const ProcessorGroup* pc,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw,
                  int device,
                  bool modifies_divQ,
                  Task::WhichDW which_abskg_dw,
                  Task::WhichDW which_sigmaT4_dw )
{
  // stub
}


