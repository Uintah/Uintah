#ifndef __OPERATORS_VECTORPARTICLESOPERATOR_H__
#define __OPERATORS_VECTORPARTICLESOPERATOR_H__

#include "VectorOperatorFunctors.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Uintah/Dataflow/Ports/VectorParticlesPort.h>
#include <Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Uintah/Core/Grid/ShareAssignParticleVariable.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
using namespace SCIRun;

  class VectorParticlesOperator: public Module {
  public:
    VectorParticlesOperator(GuiContext* ctx);
    virtual ~VectorParticlesOperator() {}
    
    virtual void execute(void);
    
  private:
    template<class VectorOp>
      void computeScalars(VectorParticles* vectors, ScalarParticles* scalars,
			  VectorOp op /* VectorOp should be a functor for
					 converting vectors scalars */ );
    //    GuiString gui_status;
    GuiInt guiOperation;

    VectorParticlesIPort *in;

    ScalarParticlesOPort *spout;
  };

template<class VectorOp>
void VectorParticlesOperator::computeScalars(VectorParticles* pTP, 
					     ScalarParticles* pSP,
					     VectorOp op
					    /* VectorOp should be a functor for
					       converting vectors scalars */ )
{
  pSP->Set(pTP->getParticleSet());
  //  pSP->Set(pTP->getLevel());
  ShareAssignParticleVariable<double> scalarSet;
 
  vector< ShareAssignParticleVariable<Vector> >& vectors = pTP->get();
  vector< ShareAssignParticleVariable<Vector> >::const_iterator iter;
  for (iter = vectors.begin(); iter != vectors.end(); iter++) {
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
#endif // __OPERATORS_VECTORPARTICLESOPERATOR_H__

