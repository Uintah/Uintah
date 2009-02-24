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


#ifndef __OPERATORS_VECTORPARTICLESOPERATOR_H__
#define __OPERATORS_VECTORPARTICLESOPERATOR_H__

#include "VectorOperatorFunctors.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Uintah/Dataflow/Ports/VectorParticlesPort.h>
#include <Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <string>

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

    // gcc 3.4 needs to instantiate a tmp
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
#endif // __OPERATORS_VECTORPARTICLESOPERATOR_H__

