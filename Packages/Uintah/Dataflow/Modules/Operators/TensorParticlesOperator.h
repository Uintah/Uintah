#ifndef __OPERATORS_TENSORPARTICLESOPERATOR_H__
#define __OPERATORS_TENSORPARTICLESOPERATOR_H__

#include "TensorOperatorFunctors.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Uintah/Dataflow/Ports/TensorParticlesPort.h>
#include <Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Uintah/Core/Grid/ShareAssignParticleVariable.h>
#include <sgi_stl_warnings_off.h>
#include <string>
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
    scalarSet = ParticleVariable<double>(subset);
    for (ParticleSubset::iterator sub_iter = subset->begin();
	 sub_iter != subset->end(); sub_iter++) {
      scalarSet[*sub_iter] = op((*iter)[*sub_iter]);
    }
    pSP->AddVar(scalarSet);
  }
}
} //end namespace Uintah
#endif // __OPERATORS_TENSORPARTICLESOPERATOR_H__

