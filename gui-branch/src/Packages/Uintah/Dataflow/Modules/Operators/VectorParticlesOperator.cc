#include "VectorParticlesOperator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Uintah/Core/Datatypes/VectorParticles.h>
#include <Uintah/Core/Datatypes/ScalarParticles.h>
#include <Uintah/Core/Grid/ParticleVariable.h>
#include <Uintah/Core/Grid/ParticleSubset.h>

//#include <SCICore/Math/Mat.h>

namespace Uintah {


extern "C" Module* make_VectorParticlesOperator( const string& id ) { 
  return scinew VectorParticlesOperator( id );
}


VectorParticlesOperator::VectorParticlesOperator(const string& id)
  : Module("VectorParticlesOperator",id,Source, "Operators", "Uintah"),
    guiOperation("operation", id, this)
{
  // Create Ports
  // Add ports to the Module
  add_iport(in);
  add_oport(spout);
}
  
void VectorParticlesOperator::execute(void) {
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  in = (VectorParticlesIPort *) get_iport("Vector Particles");
  spout = (ScalarParticlesOPort *)get_oport("Scalar Particles");

  VectorParticlesHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }

  VectorParticles* pTP = hTF.get_rep();
  ScalarParticles* pSP = scinew ScalarParticles();

  switch(guiOperation.get()) {
  case 0: // extract element
  case 1: // 2D eigen-value/vector
  case 2: // pressure
    computeScalars(pTP, pSP,
		   VectorElementExtractionOp(guiOperation.get()));
    break;
  case 3: // equivalent stress 
    computeScalars(pTP, pSP, LengthOp());
    break;
  default:
    std::cerr << "VectorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << guiOperation.get() << "\n";
  }

  spout->send(pSP);
}



} // end namespace Uintah
