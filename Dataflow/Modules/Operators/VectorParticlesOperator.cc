#include <Packages/Uintah/Dataflow/Modules/Operators/VectorParticlesOperator.h>
#include <Uintah/Core/Datatypes/VectorParticles.h>
#include <Uintah/Core/Datatypes/ScalarParticles.h>
#include <Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Uintah/Core/Grid/Variables/ParticleSubset.h>

#include <Core/Malloc/Allocator.h>

#include <math.h>

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
  ScalarParticlesHandle hSP;
  
  if(!(in->get(hTF) && hTF.get_rep())){
    error("VectorParticlesOperator::execute(void) Didn't get a handle");
    return;
  }

  VectorParticles* pTP = hTF.get_rep();
  hSP = scinew ScalarParticles();
  if (hSP.get_rep() == 0) {
    error("Error allocating ScalarParticles");
    return;
  }

  switch(guiOperation.get()) {
  case 0: // extract U
  case 1: // extract V
  case 2: // extract W
    computeScalars(pTP, hSP.get_rep(),
		   VectorElementExtractionOp(guiOperation.get()));
    break;
  case 3: // extract the length 
    computeScalars(pTP, hSP.get_rep(), LengthOp());
    break;
  case 4: // extract the curvature
    computeScalars(pTP, hSP.get_rep(), VorticityOp());
    break;
  default:
    std::cerr << "VectorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }

  spout->send(hSP);
}



} // end namespace Uintah
