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


#include "ParticleEigenEvaluator.h"
#include <cmath>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Datatypes/TensorParticles.h>
#include <Packages/Uintah/Core/Datatypes/ScalarParticles.h>
#include <Packages/Uintah/Core/Datatypes/VectorParticles.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>

namespace Uintah {

  DECLARE_MAKER(ParticleEigenEvaluator)

ParticleEigenEvaluator::ParticleEigenEvaluator(GuiContext* ctx)
  : Module("ParticleEigenEvaluator",ctx,Source, "Operators", "Uintah"),
    guiEigenSelect(get_ctx()->subVar("eigenSelect"))
    //    gui_status(get_ctx()->subVar("gui_status")),
{
}
  
void ParticleEigenEvaluator::execute(void) 
{
  //  gui_status.set("Calling EigenEvaluator!"); 

  in = (TensorParticlesIPort *) get_iport("Tensor Particles");
  spout = (ScalarParticlesOPort *) get_oport("Eigenvalue Particles");
  vpout = (VectorParticlesOPort *) get_oport("Eigenvector Particles");


  TensorParticlesHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"ParticleEigenEvaluator::execute(void) Didn't get a handle\n";
    return;
  }

  int chosenEValue = guiEigenSelect.get();

  TensorParticles* pTP = hTF.get_rep();
  ParticleVariable<double> selectedEValues;
  ParticleVariable<Vector> selectedEVectors;

  ScalarParticles* eValues = scinew ScalarParticles();
  eValues->Set(pTP->getParticleSet());
  VectorParticles* eVectors = scinew VectorParticles();
  eVectors->Set(pTP->getParticleSet());
  
  int num_eigen_values;
  const Matrix3* pM;
  double e[3];
  std::vector<Vector> eigenVectors;

  vector< ShareAssignParticleVariable<Matrix3> >& tensors = pTP->get();
  vector< ShareAssignParticleVariable<Matrix3> >::const_iterator iter;
  for (iter = tensors.begin(); iter != tensors.end(); iter++) {
    ParticleSubset* subset = (*iter).getParticleSubset();
    ParticleVariable<double> pDoubles(subset);
    ParticleVariable<Vector> pVectors(subset);
    selectedEValues.copyPointer(pDoubles);
    selectedEVectors.copyPointer(pVectors);
    for (ParticleSubset::iterator sub_iter = subset->begin();
	 sub_iter != subset->end(); sub_iter++) {
      pM = &(*iter)[*sub_iter];
      double val = 0;
      Vector vec(0, 0, 0);
      
      num_eigen_values = pM->getEigenValues(e[0], e[1], e[2]);
      if (num_eigen_values > chosenEValue) {
	val = e[chosenEValue];
	eigenVectors = pM->getEigenVectors(e[chosenEValue], e[0]);
	if (eigenVectors.size() == 1)
	  vec = eigenVectors[0].normal();
      }
      
      selectedEValues[*sub_iter] = val;
      selectedEVectors[*sub_iter] = vec;
    }
    eValues->AddVar(selectedEValues);
    eVectors->AddVar(selectedEVectors);
  }

  spout->send(eValues);
  vpout->send(eVectors);  
}

}






