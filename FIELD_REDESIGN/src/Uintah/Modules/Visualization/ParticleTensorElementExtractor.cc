#include "ParticleTensorElementExtractor.h"
#include <math.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Datatypes/TensorParticles.h>
#include <Uintah/Datatypes/ScalarParticles.h>

//#include <SCICore/Math/Mat.h>

namespace Uintah {
namespace Modules {
 
using namespace SCICore::Containers;
using namespace PSECore::Dataflow;


extern "C" Module* make_ParticleTensorElementExtractor( const clString& id ) { 
  return scinew ParticleTensorElementExtractor( id );
}

template<class TensorField, class ScalarField>
void extractElements(TensorField* tensorField, ScalarField* elementValues,
		     int row, int column);
 

ParticleTensorElementExtractor::ParticleTensorElementExtractor(const
							       clString& id)
  : Module("ParticleTensorElementExtractor",id,Source),
    tclRow("row", id, this),
    tclColumn("column", id, this)
    //    tcl_status("tcl_status", id, this),
{
  // Create Ports
  in = new TensorParticlesIPort(this, "TensorParticles");
  spout = new ScalarParticlesOPort(this, "TensorElementParticles");

  // Add ports to the Module
  add_iport(in);
  add_oport(spout);
}
  
void ParticleTensorElementExtractor::execute(void) {
  //  tcl_status.set("Calling EigenEvaluator!"); 
  TensorParticlesHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }

  int row = tclRow.get();
  int column = tclColumn.get();
  
  TensorParticles* pTP = hTF.get_rep();
  ParticleVariable<double> selectedElements;

  ScalarParticles* elements = scinew ScalarParticles();
  elements->Set(pTP->getParticleSet());

  vector< ParticleVariable<Matrix3> >& tensors = pTP->get();
  vector< ParticleVariable<Matrix3> >::const_iterator iter;
  for (iter = tensors.begin(); iter != tensors.end(); iter++) {
    ParticleSubset* subset = (*iter).getParticleSubset();
    selectedElements = ParticleVariable<double>(subset);
    for (ParticleSubset::iterator sub_iter = subset->begin();
	 sub_iter != subset->end(); sub_iter++) {
      selectedElements[*sub_iter] = (*iter)[*sub_iter](row, column);
    }
    elements->AddVar(selectedElements);
  }

  spout->send(elements);
}

}
}


