#include "VectorParticlesOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Uintah/Core/Datatypes/VectorParticles.h>
#include <Uintah/Core/Datatypes/ScalarParticles.h>
#include <Uintah/Core/Grid/ParticleVariable.h>
#include <Uintah/Core/Grid/ParticleSubset.h>

//#include <SCICore/Math/Mat.h>

namespace Uintah {


 DECLARE_MAKER(VectorParticlesOperator)


VectorParticlesOperator::VectorParticlesOperator(GuiContext* ctx)
  : Module("VectorParticlesOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(ctx->subVar("operation"))
{
}
  
void VectorParticlesOperator::execute(void) 
{
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  in = (VectorParticlesIPort *) get_iport("Vector Particles");
  spout = (ScalarParticlesOPort *)get_oport("Scalar Particles");

  VectorParticlesHandle hTF;
  
  if(!in->get(hTF)){
    error("VectorParticlesOperator::execute(void) Didn't get a handle");
    return;
  }

  VectorParticles* pTP = hTF.get_rep();
  ScalarParticles* pSP = scinew ScalarParticles();

  switch(guiOperation.get()) {
  case 0: // extract U
  case 1: // extract V
  case 2: // extract W
    computeScalars(pTP, pSP,
		   VectorElementExtractionOp(guiOperation.get()));
    break;
  case 3: // extract the length 
    computeScalars(pTP, pSP, LengthOp());
    break;
  case 4: // extract the curvature
    computeScalars(pTP, pSP, VorticityOp());
    break;
  default:
    std::cerr << "VectorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }

  spout->send(pSP);
}



} // end namespace Uintah
