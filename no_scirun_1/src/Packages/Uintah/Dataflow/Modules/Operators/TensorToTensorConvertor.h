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


#ifndef __OPERATORS_TENSORTOTENSORCONVERTOR_H__
#define __OPERATORS_TENSORTOTENSORCONVERTOR_H__

#include "TensorOperatorFunctors.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Uintah/Dataflow/Ports/TensorParticlesPort.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace Uintah {
using namespace SCIRun;


  class TensorToTensorConvertor : public Module {
  public:
    TensorToTensorConvertor(GuiContext*);

    virtual ~TensorToTensorConvertor();

    virtual void execute(void);

  private:

    template<class TensorOp>
      void computeTensor(TensorParticles* pInput, TensorParticles* pOutput,
			 TensorOp op);
    GuiInt guiOperation;
    TensorParticlesIPort *inPort;
    TensorParticlesOPort *outPort;
  
  };

  template<class TensorOp>
  void TensorToTensorConvertor::computeTensor(TensorParticles* pInput, 
					      TensorParticles* pOutput,
					      TensorOp op)
  {
    // Get the particle set
    pOutput->Set(pInput->getParticleSet());
    ShareAssignParticleVariable<Matrix3> outSet;
    
    vector<ShareAssignParticleVariable<Matrix3> >& inTensor = pInput->get();
    vector<ShareAssignParticleVariable<Matrix3> >::const_iterator iter;
    
    for (iter = inTensor.begin(); iter != inTensor.end(); iter++) {
      
      ParticleSubset* subset = (*iter).getParticleSubset();
      ParticleVariable<Matrix3> tmp(subset);
      outSet = tmp;
      for (ParticleSubset::iterator sub_iter = subset->begin();
	   sub_iter != subset->end(); sub_iter++) {
	outSet[*sub_iter] = op((*iter)[*sub_iter]);
      }
      pOutput->AddVar(outSet);
    }
  }
  
} //end namespace Uintah

#endif 
  
