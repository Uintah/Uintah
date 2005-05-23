#ifndef __OPERATORS_TENSORTOTENSORCONVERTOR_H__
#define __OPERATORS_TENSORTOTENSORCONVERTOR_H__

#include "TensorOperatorFunctors.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Uintah/Dataflow/Ports/TensorParticlesPort.h>
#include <Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>
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
  
