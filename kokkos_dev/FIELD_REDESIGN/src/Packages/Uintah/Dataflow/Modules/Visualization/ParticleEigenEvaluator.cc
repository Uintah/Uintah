#include "ParticleEigenEvaluator.h"
#include <math.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Datatypes/TensorParticles.h>
#include <Uintah/Datatypes/ScalarParticles.h>
#include <Uintah/Datatypes/VectorParticles.h>

namespace Uintah {
namespace Modules {
 
using namespace SCICore::Containers;
using namespace PSECore::Dataflow;


extern "C" Module* make_ParticleEigenEvaluator( const clString& id ) { 
  return scinew ParticleEigenEvaluator( id );
}

ParticleEigenEvaluator::ParticleEigenEvaluator(const clString& id)
  : Module("ParticleEigenEvaluator",id,Source),
    tclEigenSelect("eigenSelect", id, this)
    //    tcl_status("tcl_status", id, this),
{
  // Create Ports
  in = scinew TensorParticlesIPort(this, "TensorParticles");
  spout = scinew ScalarParticlesOPort(this, "EigenValueParticles");
  vpout = scinew VectorParticlesOPort(this, "EigenVectorParticles");

  // Add ports to the Module
  add_iport(in);
  add_oport(spout);
  add_oport(vpout);
}
  
void ParticleEigenEvaluator::execute(void) {
  //  tcl_status.set("Calling EigenEvaluator!"); 
  TensorParticlesHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }

  int chosenEValue = tclEigenSelect.get();

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

  vector< ParticleVariable<Matrix3> >& tensors = pTP->get();
  vector< ParticleVariable<Matrix3> >::const_iterator iter;
  for (iter = tensors.begin(); iter != tensors.end(); iter++) {
    ParticleSubset* subset = (*iter).getParticleSubset();
    selectedEValues = ParticleVariable<double>(subset);
    selectedEVectors = ParticleVariable<Vector>(subset);
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
}





