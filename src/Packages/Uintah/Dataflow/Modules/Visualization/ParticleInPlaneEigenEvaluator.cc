#include "ParticleInPlaneEigenEvaluator.h"
#include <math.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Datatypes/TensorParticles.h>
#include <Uintah/Datatypes/ScalarParticles.h>
#include <Uintah/Datatypes/VectorParticles.h>
//#include <SCICore/Math/Mat.h>

namespace Uintah {
namespace Modules {
 
using namespace SCICore::Containers;
using namespace PSECore::Dataflow;


extern "C" Module* make_ParticleInPlaneEigenEvaluator( const clString& id ) { 
  return scinew ParticleInPlaneEigenEvaluator( id );
}

typedef int (Matrix3::*pmfnPlaneEigenFunc)(double& e1, double& e2) const;
extern pmfnPlaneEigenFunc planeEigenValueFuncs[3];
  
ParticleInPlaneEigenEvaluator::ParticleInPlaneEigenEvaluator(const
							     clString& id)
  : Module("ParticleInPlaneEigenEvaluator",id,Source),
    tclPlaneSelect("planeSelect", id, this),
    tclCalculationType("calcType", id, this),
    tclDelta("delta", id, this)
    //    tcl_status("tcl_status", id, this),
{
  // Create Ports
  in = new TensorParticlesIPort(this, "TensorParticles");
  spout = new ScalarParticlesOPort(this, "EigenDataParticles");

  // Add ports to the Module
  add_iport(in);
  add_oport(spout);

  planeEigenValueFuncs[0] = &Matrix3::getYZEigenValues;
  planeEigenValueFuncs[1] = &Matrix3::getXZEigenValues;
  planeEigenValueFuncs[2] = &Matrix3::getXYEigenValues;
}
  
void ParticleInPlaneEigenEvaluator::execute(void) {
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  TensorParticlesHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }

  int chosenPlane = tclPlaneSelect.get();
  bool takeSin = (tclCalculationType.get() != 0);
  double delta =  tclDelta.get();

  TensorParticles* pTP = hTF.get_rep();
  ParticleVariable<double> eDataSet;

  ScalarParticles* eData = scinew ScalarParticles();
  eData->Set(pTP->getParticleSet());
  
  int num_eigen_values;
  const Matrix3* pM;
  double e1, e2;

  vector< ParticleVariable<Matrix3> >& tensors = pTP->get();
  vector< ParticleVariable<Matrix3> >::const_iterator iter;
  for (iter = tensors.begin(); iter != tensors.end(); iter++) {
    ParticleSubset* subset = (*iter).getParticleSubset();
    eDataSet = ParticleVariable<double>(subset);
    for (ParticleSubset::iterator sub_iter = subset->begin();
	 sub_iter != subset->end(); sub_iter++) {
      pM = &(*iter)[*sub_iter];
      double val = 0;

      num_eigen_values = (pM->*planeEigenValueFuncs[chosenPlane])(e1, e2);

      // There are either two equivalent eigen values or they are
      // imaginary numbers.  Either case, just use 0 as the diff.
      if (num_eigen_values == 2)
	val = (takeSin ? sin((e1 - e2) / delta) : (e1 - e2)); // e1 > e2
      
      eDataSet[*sub_iter] = val;
    }
    eData->AddVar(eDataSet);
  }

  spout->send(eData);
}

}
}



