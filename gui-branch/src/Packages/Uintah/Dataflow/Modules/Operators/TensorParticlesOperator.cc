#include "TensorParticlesOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Uintah/Core/Datatypes/TensorParticles.h>
#include <Uintah/Core/Datatypes/ScalarParticles.h>
#include <Uintah/Core/Grid/ParticleVariable.h>
#include <Uintah/Core/Grid/ParticleSubset.h>

//#include <SCICore/Math/Mat.h>

namespace Uintah {


extern "C" Module* make_TensorParticlesOperator( const string& id ) { 
  return scinew TensorParticlesOperator( id );
}


TensorParticlesOperator::TensorParticlesOperator(const string& id)
  : Module("TensorParticlesOperator",id,Source, "Operators", "Uintah"),
    guiOperation("operation", id, this),
    guiRow("row", id, this),
    guiColumn("column", id, this),
    guiPlaneSelect("planeSelect", id, this),
    guiDelta("delta", id, this),
    guiEigen2DCalcType("eigen2D-calc-type", id, this)
    //    tcl_status("tcl_status", id, this),
{
}
  
void TensorParticlesOperator::execute(void) {
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
 
  in = ( TensorParticlesIPort *) get_iport("Tensor Particles");
  spout = ( ScalarParticlesOPort *) get_oport("Scalar Particles");

  TensorParticlesHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }

  TensorParticles* pTP = hTF.get_rep();
  ScalarParticles* pSP = scinew ScalarParticles();

  switch(guiOperation.get()) {
  case 0: // extract element
    computeScalars(pTP, pSP,
		   TensorElementExtractionOp(guiRow.get(), guiColumn.get()));
    break;
  case 1: // 2D eigen-value/vector
    if (guiEigen2DCalcType.get() == 0) {
      // e1 - e2
      int plane = guiPlaneSelect.get();
      if (plane == 2)
	computeScalars(pTP, pSP, Eigen2DXYOp());
      else if (plane == 1)
	computeScalars(pTP, pSP, Eigen2DXZOp());
      else
	computeScalars(pTP, pSP, Eigen2DYZOp());
    }
    else {
      // cos(e1 - e2) / delta
      int plane = guiPlaneSelect.get();
      double delta = guiDelta.get();
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
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }

  spout->send(pSP);
}




} // end namespace Uintah
