#include "TensorToTensorConvertor.h"
#include <Uintah/Core/Datatypes/TensorParticles.h>

//#include <SCICore/Math/Mat.h>

namespace Uintah {


DECLARE_MAKER(TensorToTensorConvertor)
TensorToTensorConvertor::TensorToTensorConvertor(GuiContext* ctx)
  : Module("TensorToTensorConvertor", ctx, Source, "Operators", "Uintah"),
  guiOperation(ctx->subVar("operation"))
{
}

TensorToTensorConvertor::~TensorToTensorConvertor()
{
}

void
 TensorToTensorConvertor::execute(void)
{
  // Set up ports
  inPort = (TensorParticlesIPort*) get_iport("Input Tensor");
  outPort = (TensorParticlesOPort*) get_oport("Output Tensor");

  // Set up handle
  TensorParticlesHandle handle;
  if (!inPort->get(handle)) {
    error("TensorToTensorConvertor::execute(void) No input handle found.");
    return;
  }

  // Get the data 
  TensorParticles* pInput = handle.get_rep();

  // Set up the output pointer
  TensorParticles* pOutput = scinew TensorParticles();

  // Find selection and execute conversions
  switch(guiOperation.get()) {
    case 0:
      computeTensor(pInput, pOutput, NullTensorOp());
      break;
    case 1:
      computeTensor(pInput, pOutput, GreenLagrangeStrainTensorOp());
      break;
    case 2:
      computeTensor(pInput, pOutput, CauchyGreenDeformationTensorOp());
      break;
    case 3:
      computeTensor(pInput, pOutput, FingerDeformationTensorOp());
      break;
    default:
      std::cerr << "TensorToTensorConvertor::execute(void): "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }

  outPort->send(pOutput);
}

} // End namespace Uintah


