/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include "TensorToTensorConvertor.h"
#include <Uintah/Core/Datatypes/TensorParticles.h>

//#include <SCICore/Math/Mat.h>

namespace Uintah {


DECLARE_MAKER(TensorToTensorConvertor)
TensorToTensorConvertor::TensorToTensorConvertor(GuiContext* ctx)
  : Module("TensorToTensorConvertor", ctx, Source, "Operators", "Uintah"),
  guiOperation(get_ctx()->subVar("operation"))
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


