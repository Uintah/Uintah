#include "TensorParticlesOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Uintah/Core/Datatypes/TensorParticles.h>
#include <Uintah/Core/Datatypes/ScalarParticles.h>
#include <Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Uintah/Core/Grid/Variables/ParticleSubset.h>

//#include <SCICore/Math/Mat.h>

namespace Uintah {


  DECLARE_MAKER(TensorParticlesOperator)

TensorParticlesOperator::TensorParticlesOperator(GuiContext* ctx)
  : Module("TensorParticlesOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(ctx->subVar("operation")),
    guiRow(ctx->subVar("row")),
    guiColumn(ctx->subVar("column")),
    guiPlaneSelect(ctx->subVar("planeSelect")),
    guiDelta(ctx->subVar("delta")),
    guiEigen2DCalcType(ctx->subVar("eigen2D-calc-type")),
    guiNx(ctx->subVar("nx")),
    guiNy(ctx->subVar("ny")),
    guiNz(ctx->subVar("nz")),
    guiTx(ctx->subVar("tx")),
    guiTy(ctx->subVar("ty")),
    guiTz(ctx->subVar("tz"))
    //    tcl_status(ctx->subVar("tcl_status")),
{
}
  
void TensorParticlesOperator::execute(void) 
{
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
 
  in = ( TensorParticlesIPort *) get_iport("Tensor Particles");
  spout = ( ScalarParticlesOPort *) get_oport("Scalar Particles");

  TensorParticlesHandle hTF;
  
  if(!in->get(hTF)){
    error("TensorParticlesOperator::execute(void) Didn't get a handle.");
    return;
  }

  TensorParticles* pTP = hTF.get_rep();
  ScalarParticles* pSP = scinew ScalarParticles();

  switch(guiOperation.get()) {
  case 0: // extract element
    computeScalars(pTP, pSP,
		   TensorElementExtractionOp(guiRow.get(), guiColumn.get()));
    break;
  case 1: // 2D eigen-value/vector
    if (guiEigen2DCalcType.get() == 0) {
      // e1 - e2
      int plane = guiPlaneSelect.get();
      if (plane == 2)
	computeScalars(pTP, pSP, Eigen2DXYOp());
      else if (plane == 1)
	computeScalars(pTP, pSP, Eigen2DXZOp());
      else
	computeScalars(pTP, pSP, Eigen2DYZOp());
    }
    else {
      // cos(e1 - e2) / delta
      int plane = guiPlaneSelect.get();
      double delta = guiDelta.get();
      if (plane == 2)
	computeScalars(pTP, pSP, Eigen2DXYCosOp(delta));
      else if (plane == 1)
	computeScalars(pTP, pSP, Eigen2DXZCosOp(delta));
      else
	computeScalars(pTP, pSP, Eigen2DYZCosOp(delta));
    }
    break;
  case 2: // pressure
    computeScalars(pTP, pSP, PressureOp());
    break;
  case 3: // equivalent stress 
    computeScalars(pTP, pSP, EquivalentStressOp());
    break;
  case 4: // Octahedral shear stress
    computeScalars(pTP, pSP, OctShearStressOp());
    break;
  case 5: // n . sigma. t
    {
    double nx = guiNx.get(); double ny = guiNy.get(); double nz = guiNz.get();
    double tx = guiTx.get(); double ty = guiTy.get(); double tz = guiTz.get();
    computeScalars(pTP, pSP, NDotSigmaDotTOp(nx, ny, nz, tx, ty, tz));
    }
    break;
  default:
    std::cerr << "TensorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }

  spout->send(pSP);
}




} // end namespace Uintah
