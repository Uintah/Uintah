/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#include <Core/Util/DebugStream.h>
#include "HypoViscoElasticDevStress.h"
#include <Core/Exceptions/ProblemSetupException.h>

using namespace std;
using namespace Uintah;
static DebugStream dbg("HypoViscoElasticDevStress", false);

HypoViscoElasticDevStress::HypoViscoElasticDevStress(ProblemSpecP& ps)
{  
  ps->get("tau", d_tau_MW);
  ps->get("mu",  d_mu_MW);

  // bulletproofing
  if( d_tau_MW.size() != d_mu_MW.size() ){
    throw ProblemSetupException("ERROR:  The number of maxwell elements for tau != mu", __FILE__, __LINE__);
  }

  if( d_tau_MW.size() == 0 || d_mu_MW.size() == 0 ){
    throw ProblemSetupException("ERROR:  The number of maxwell elements for tau & mu > 0", __FILE__, __LINE__);
  }

  // for speed:
  for (unsigned int j = 0; j< d_tau_MW.size(); j++){
    d_inv_tau_MW.push_back( 1.0/d_tau_MW[j] );
  }

  // number of Maxwell Elements
  d_MaxwellElements = d_tau_MW.size();
  
  // create labels for each maxwell element
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
     ostringstream name, name2;
     name  << "sigmaDev"<<j;
     name2 << name.str() << "+";

    // create internal variable labels
    d_sigmaDevLabel.push_back(          VarLabel::create(name.str(),   ParticleVariable<Matrix3>::getTypeDescription()) );
    d_sigmaDevLabel_preReloc.push_back( VarLabel::create(name2.str(),  ParticleVariable<Matrix3>::getTypeDescription()) );
  }
  
  // This is a std::vector of ParticleVariable<Matrix3>
  // Each particle needs d_MaxwellElements matrices 
  d_sigmaDev.resize(d_MaxwellElements);
  d_sigmaDev_new.resize(d_MaxwellElements);
  
  // create the arrays
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    d_sigmaDev[j]     = scinew constParticleVariable<Matrix3> () ;
    d_sigmaDev_new[j] = scinew ParticleVariable<Matrix3> () ; 
  }
}
//______________________________________________________________________
//
HypoViscoElasticDevStress::~HypoViscoElasticDevStress()
{
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    VarLabel::destroy( d_sigmaDevLabel[j] );
    VarLabel::destroy( d_sigmaDevLabel_preReloc[j] );
    delete d_sigmaDev[j];
    delete d_sigmaDev_new[j];
  }
}
//______________________________________________________________________
//
void HypoViscoElasticDevStress::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP flow_ps = ps->appendChild( "deviatoric_stress_model" );
  flow_ps->setAttribute( "type","hypoViscoElastic" );
  flow_ps->appendElement( "tau", d_tau_MW );
  flow_ps->appendElement( "mu",  d_mu_MW );
}
//______________________________________________________________________
//   
void 
HypoViscoElasticDevStress::addInitialComputesAndRequires(Task* task,
                                                         const MPMMaterial* matl)
{
  dbg << " hypoViscoElastic::addInitialComputesAndRequires " << endl;
  const MaterialSubset* matlset = matl->thisMaterial();
  
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    task->computes( d_sigmaDevLabel[j], matlset );
  }
}
//______________________________________________________________________
//    Called by computeStressTensor()
void 
HypoViscoElasticDevStress::addComputesAndRequires(Task* task,
                                                  const MPMMaterial* matl)
{
  dbg << " hypoViscoElastic:addComputesAndRequires 1 " << endl;
  const MaterialSubset* matlset = matl->thisMaterial();
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    
    task->requires( Task::OldDW, d_sigmaDevLabel[j], matlset,Ghost::None );
    task->computes( d_sigmaDevLabel_preReloc[j], matlset );
  }
}
//______________________________________________________________________
//    Called by computeStressTensorImplicit
void
HypoViscoElasticDevStress::addComputesAndRequires(Task* task,
                                                  const MPMMaterial* matl,
                                                  bool SchedParent)
{

  dbg << " hypoViscoElastic:addComputesAndRequires 2 " << endl;
  const MaterialSubset* matlset = matl->thisMaterial();
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    if(SchedParent){
      task->requires( Task::ParentOldDW, d_sigmaDevLabel[j], matlset,Ghost::None );
    }else{
      task->requires( Task::OldDW,       d_sigmaDevLabel[j], matlset,Ghost::None );
    }
  }
}
//______________________________________________________________________
//
void 
HypoViscoElasticDevStress::addParticleState(std::vector<const VarLabel*>& from,
                                            std::vector<const VarLabel*>& to)
{
  dbg << " hypoViscoElastic:addParticleState " << endl;
  for( unsigned int j = 0; j< d_MaxwellElements; j++){  
    from.push_back(d_sigmaDevLabel[j]);
    to.push_back(d_sigmaDevLabel_preReloc[j]);
  }
}

//______________________________________________________________________
//
void 
HypoViscoElasticDevStress::initializeInternalVars(ParticleSubset* pset,
                                                  DataWarehouse* new_dw)
{
  dbg << " hypoViscoElastic:initializeInternalVars " << endl;
  
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    
    ParticleVariable<Matrix3> sigmaDev_new;
    new_dw->allocateAndPut( sigmaDev_new, d_sigmaDevLabel[j], pset );
    Matrix3 zero(0);
  
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end(); iter++) {
      sigmaDev_new[*iter] = zero;
    }
  }
}
//______________________________________________________________________
//
void 
HypoViscoElasticDevStress::getInternalVars(ParticleSubset* pset,
                                           DataWarehouse* old_dw) 
{
  dbg << " hypoViscoElastic:getInternalVars " << endl;
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    old_dw->get( *d_sigmaDev[j], d_sigmaDevLabel[j], pset );
  }
}
//______________________________________________________________________
//
void 
HypoViscoElasticDevStress::allocateAndPutInternalVars(ParticleSubset* pset,
                                                      DataWarehouse* new_dw) 
{
  dbg << " hypoViscoElastic:allocateAndPutInternalVars " << endl;
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    new_dw->allocateAndPut( *d_sigmaDev_new[j], d_sigmaDevLabel_preReloc[j], pset );
  }
}

//______________________________________________________________________
//  Initializing to zero for the sake of RigidMPM's carryForward
void
HypoViscoElasticDevStress::allocateAndPutRigid(ParticleSubset* pset,
                                               DataWarehouse* new_dw)
{
  dbg << " hypoViscoElastic:allocateAndPutRigid " << endl;
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    ParticleVariable<Matrix3> sigmaDev_new;
    
    new_dw->allocateAndPut( sigmaDev_new, d_sigmaDevLabel_preReloc[j], pset );
    
    Matrix3 zero(0);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end(); iter++){
       sigmaDev_new[*iter] = zero;
    }
  }
}


//______________________________________________________________________
///
void HypoViscoElasticDevStress::computeDeviatoricStressInc( const particleIndex idx,
                                                            const PlasticityState* plaState,
                                                            DeformationState* defState,
                                                            const double delT ){ 

  dbg << " hypoViscoElastic:computeDevStessInc " << endl;

  double mu = plaState->shearModulus;  // WARNING THIS MAY NOT BE SUM(d_mu_MW) 
                                       // other routines modify this.
   
  
  Matrix3 sigmadot = 2.0 * mu * defState->tensorEta;

  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    sigmadot -= ( *d_sigmaDev[j] )[idx] * d_inv_tau_MW[j];
  }

  //sigma_dev_trial = sigma_dev_n + sigmadot*delT;    (original Equation.)

  defState->devStressInc = sigmadot*delT; 

  // bulletproofing
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    if( d_tau_MW[j] < delT ){
       ostringstream warn;
       warn<< "ERROR: hypoViscoElastic:computeDevStessInc \n"
           << "tau ["<< d_tau_MW[j] << "] < delT ["<< delT <<"] ";
       throw InternalError("warn.str()",__FILE__,__LINE__);
    }
  }
}

//______________________________________________________________________
//
void HypoViscoElasticDevStress::updateInternalStresses( const particleIndex idx,
                                                        const Matrix3& dp,
                                                        DeformationState* defState,
                                                        const double delT ){
  dbg << " hypoViscoElastic:updateInternalStresses " << endl; 
  const Matrix3 tensorEta = defState->tensorEta;
  double A = 0.0;

  for( unsigned int j = 0; j< d_MaxwellElements; j++){

    ( *d_sigmaDev_new[j] )[idx] = ( *d_sigmaDev[j] )[idx] + (  2.0 * d_mu_MW[j] * (tensorEta - dp)  - ( *d_sigmaDev[j] )[idx] * d_inv_tau_MW[j] ) * delT;
    //^^^^^^^^^^^^^^^^^^
    //  I don't like this notation -Todd
    //cout << "    d_sigmaDev_new " << ( *d_sigmaDev_new[j] )[idx] << " d_sigmaDev " << ( *d_sigmaDev[j] )[idx] << endl;
    
    double B = ( *d_sigmaDev_new[j] )[idx].NormSquared();
    A += B * d_inv_tau_MW[j]/(2.0*d_mu_MW[j] );
  }
  defState->viscoElasticWorkRate = A;
}

//______________________________________________________________________
//
void HypoViscoElasticDevStress::rotateInternalStresses( const particleIndex idx,
                                                        const Matrix3& tensorR){ 
  dbg << " hypoViscoElastic:rotateInternalStresses " << endl;
  
  for( unsigned int j = 0; j< d_MaxwellElements; j++){
    ( *d_sigmaDev_new[j] )[idx] = tensorR.Transpose() * ( ( *d_sigmaDev_new[j] )[idx] * tensorR );
    //cout << "    d_sigmaDev_new " << ( *d_sigmaDev_new[j] )[idx] << " d_sigmaDev " << ( *d_sigmaDev[j] )[idx] << endl;
  }
}
