#include "ParticleEigenEvaluator.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Uintah/Core/Datatypes/TensorParticles.h>
#include <Uintah/Core/Datatypes/ScalarParticles.h>
#include <Uintah/Core/Datatypes/VectorParticles.h>

namespace Uintah {

extern "C" Module* make_ParticleEigenEvaluator( const string& id ) { 
  return scinew ParticleEigenEvaluator( id );
}

ParticleEigenEvaluator::ParticleEigenEvaluator(const string& id)
  : Module("ParticleEigenEvaluator",id,Source),
    guiEigenSelect("eigenSelect", id, this)
    //    gui_status("gui_status", id, this),
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
  //  gui_status.set("Calling EigenEvaluator!"); 
  TensorParticlesHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }

  int chosenEValue = guiEigenSelect.get();

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
    selectedEValues.copyPointer(ParticleVariable<double>(subset));
    selectedEVectors.copyPointer(ParticleVariable<Vector>(subset));
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






