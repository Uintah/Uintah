#include "TensorParticlesOperator.h"
#include "TensorOperatorFunctors.h"
#include <math.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Datatypes/TensorField.h>
#include <Uintah/Datatypes/NCTensorField.h>
#include <Uintah/Datatypes/CCTensorField.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGCC.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/VectorFieldRGCC.h>
//#include <SCICore/Math/Mat.h>

namespace Uintah {
namespace Modules {
 
using namespace SCICore::Containers;
using namespace PSECore::Dataflow;


extern "C" Module* make_TensorParticlesOperator( const clString& id ) { 
  return scinew TensorParticlesOperator( id );
}

template<class TensorOp>
void computeScalars(TensorParticles* tensors, ScalarParticles* scalars,
		    TensorOp op /* TensorOp should be a functor for
				   converting tensors scalars */ );

TensorParticlesOperator::TensorParticlesOperator(const clString& id)
  : Module("TensorParticlesOperator",id,Source),
    tclOperation("operation", id, this),
    tclRow("row", id, this),
    tclColumn("column", id, this),
    tclPlaneSelect("planeSelect", id, this),
    tclDelta("delta", id, this),
    tclEigen2DCalcType("eigen2D-calc-type", id, this)
    //    tcl_status("tcl_status", id, this),
{
  // Create Ports
  in = scinew TensorParticlesIPort(this, "TensorField");
  spout = scinew ScalarParticlesOPort(this, "ScalarField");

  // Add ports to the Module
  add_iport(in);
  add_oport(spout);
}
  
void TensorParticlesOperator::execute(void) {
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  TensorParticlesHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }

  TensorParticles* pTP = hTF.get_rep();
  ScalarParticles* pSP = scinew ScalarParticles();

  switch(tclOperation.get()) {
  case 0: // extract element
    computeScalars(pTP, pSP,
		   TensorElementExtractionOp(tclRow.get(), tclColumn.get()));
    break;
  case 1: // 2D eigen-value/vector
    if (tclEigen2DCalcType.get() == 0) {
      // e1 - e2
      int plane = tclPlaneSelect.get();
      if (plane == 2)
	computeScalars(pTP, pSP, Eigen2DXYOp());
      else if (plane == 1)
	computeScalars(pTP, pSP, Eigen2DXZOp());
      else
	computeScalars(pTP, pSP, Eigen2DYZOp());
    }
    else {
      // cos(e1 - e2) / delta
      int plane = tclPlaneSelect.get();
      double delta = tclDelta.get();
      if (plane == 2)
	computeScalars(pTP, pSP, Eigen2DXYCosOp(delta));
      else if (plane == 1)
	computeScalars(pTP, pSP, Eigen2DXZCosOp(delta));
      else
	computeScalars(pTP, pSP, Eigen2DYZCosOp(delta));
    }
    break;
  case 2: // pressure
    computeScalars(pTP, pSP, PressureOp());
    break;
  case 3: // equivalent stress 
    computeScalars(pTP, pSP, EquivalentStressOp());
    break;
  default:
    std::cerr << "TensorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << tclOperation.get() << "\n";
  }

  spout->send(pSP);
}

template<class TensorOp>
void computeScalars(TensorParticles* pTP, ScalarParticles* pSP,
		    TensorOp op /* TensorOp should be a functor for
				    converting tensors scalars */ )
{
  Matrix3 M;
  pSP->Set(pTP->getParticleSet());
  ParticleVariable<double> scalarSet;
 
  vector< ParticleVariable<Matrix3> >& tensors = pTP->get();
  vector< ParticleVariable<Matrix3> >::const_iterator iter;
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

}
}
