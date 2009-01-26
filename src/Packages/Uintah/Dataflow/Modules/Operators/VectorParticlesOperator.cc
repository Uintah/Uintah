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


#include <Packages/Uintah/Dataflow/Modules/Operators/VectorParticlesOperator.h>
#include <Packages/Uintah/Core/Datatypes/VectorParticles.h>
#include <Packages/Uintah/Core/Datatypes/ScalarParticles.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSubset.h>

#include <Core/Malloc/Allocator.h>

#include <cmath>

//#include <SCICore/Math/Mat.h>

namespace Uintah {


 DECLARE_MAKER(VectorParticlesOperator)


VectorParticlesOperator::VectorParticlesOperator(GuiContext* ctx)
  : Module("VectorParticlesOperator",ctx,Source, "Operators", "Uintah"),
    guiOperation(get_ctx()->subVar("operation"))
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
