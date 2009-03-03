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


#ifndef __OPERATORS_TENSORPARTICLESOPERATOR_H__
#define __OPERATORS_TENSORPARTICLESOPERATOR_H__

#include <Packages/Uintah/Dataflow/Modules/Operators/TensorOperatorFunctors.h>

#include <Packages/Uintah/Dataflow/Ports/TensorParticlesPort.h>
#include <Packages/Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>

#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>

#include <sgi_stl_warnings_off.h>
#include   <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
using namespace SCIRun;

  class TensorParticlesOperator: public Module {
  public:
    TensorParticlesOperator(GuiContext* ctx);
    virtual ~TensorParticlesOperator() {}
    
    virtual void execute(void);
    
  private:
    template<class TensorOp>
      void computeScalars(TensorParticles* tensors, ScalarParticles* scalars,
			  TensorOp op /* TensorOp should be a functor for
					 converting tensors scalars */ );
    //    GuiString gui_status;
    GuiInt guiOperation;

    // element extractor operation
    GuiInt guiRow;
    GuiInt guiColumn;
    
    // eigen value/vector operation
    //GuiInt guiEigenSelect;

    // eigen 2D operation
    GuiInt guiPlaneSelect;
    GuiDouble guiDelta;
    GuiInt guiEigen2DCalcType;
    
    // n . sigma . t operation
    GuiDouble guiNx, guiNy, guiNz;
    GuiDouble guiTx, guiTy, guiTz;
    TensorParticlesIPort *in;

    ScalarParticlesOPort *spout;
  };


template<class TensorOp>
void TensorParticlesOperator::computeScalars(TensorParticles* pTP,
					     ScalarParticles* pSP,
					     TensorOp op
					  /* TensorOp should be a functor for
					     converting tensors scalars */ )
{
  Matrix3 M;
  pSP->Set(pTP->getParticleSet());
  //  pSP->Set(pTP->getLevel());
  ShareAssignParticleVariable<double> scalarSet;
 
  vector< ShareAssignParticleVariable<Matrix3> >& tensors = pTP->get();
  vector< ShareAssignParticleVariable<Matrix3> >::const_iterator iter;
  for (iter = tensors.begin(); iter != tensors.end(); iter++) {
    ParticleSubset* subset = (*iter).getParticleSubset();
    ParticleVariable<double> tmp(subset);
    scalarSet = tmp;
    for (ParticleSubset::iterator sub_iter = subset->begin();
	 sub_iter != subset->end(); sub_iter++) {
      scalarSet[*sub_iter] = op((*iter)[*sub_iter]);
    }
    pSP->AddVar(scalarSet);
  }
}
} //end namespace Uintah
#endif // __OPERATORS_TENSORPARTICLESOPERATOR_H__

