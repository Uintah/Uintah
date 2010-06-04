/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


#include <CCA/Components/MPM/ConstitutiveModel/UCNH.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;


UCNH::UCNH(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  d_useModifiedEOS = false;
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  ps->get("useModifiedEOS",d_useModifiedEOS); 
  
  // Plasticity
  ps->getWithDefault("usePlasticity", d_usePlasticity, false);
  if(d_usePlasticity)
  {
    ps->getWithDefault("alpha",d_initialData.Alpha,0.0);
    ps->require("yield_stress",d_initialData.FlowStress);
    ps->require("hardening_modulus",d_initialData.K);
      
    p_statedata_label = VarLabel::create("p.statedata_cnhp",
                                         ParticleVariable<StateData>::getTypeDescription());
    p_statedata_label_preReloc = VarLabel::create("p.statedata_cnhp+",
                                                  ParticleVariable<StateData>::getTypeDescription());
  }
  
  // Damage
  ps->getWithDefault("useDamage", d_useDamage, false);
  if(d_useDamage)
  {
    
  }
  
  // Universal Labels
  bElBarLabel = VarLabel::create("p.bElBar",
                                 ParticleVariable<Matrix3>::getTypeDescription());
  bElBarLabel_preReloc = VarLabel::create("p.bElBar+",
                                        ParticleVariable<Matrix3>::getTypeDescription());



  cout << "Plasticity: " << d_usePlasticity << endl;
  cout << "Damage: " << d_useDamage << endl << endl;    
}

UCNH::UCNH(const UCNH* cm) : ConstitutiveModel(cm)
{
  d_useModifiedEOS = cm->d_useModifiedEOS ;
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
  
  // Plasticity
  d_usePlasticity = cm->d_usePlasticity;
  if(d_usePlasticity)
  {
    d_initialData.FlowStress = cm->d_initialData.FlowStress;
    d_initialData.K = cm->d_initialData.K;
    d_initialData.Alpha = cm->d_initialData.Alpha;
    
    p_statedata_label = VarLabel::create("p.statedata_cnhp",
                                            ParticleVariable<StateData>::getTypeDescription());
    p_statedata_label_preReloc = VarLabel::create("p.statedata_cnhp+",
                                                    ParticleVariable<StateData>::getTypeDescription());
  }
  
  // Damage
  d_useDamage = cm->d_useDamage;
  if(d_useDamage)
  {
    
  }
  
  // Universal Labels
  bElBarLabel = VarLabel::create("p.bElBar",
                                 ParticleVariable<Matrix3>::getTypeDescription());
  bElBarLabel_preReloc = VarLabel::create("p.bElBar+",
                                          ParticleVariable<Matrix3>::getTypeDescription());
}

void UCNH::addParticleState(std::vector<const VarLabel*>& from,
                                       std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  // Plasticity
  if(d_usePlasticity)
  {
    from.push_back(p_statedata_label);
    from.push_back(bElBarLabel);
    to.push_back(p_statedata_label_preReloc);
    to.push_back(bElBarLabel_preReloc);
  }
  
  // Damage
  if(d_useDamage)
  {
    
  }
  
  // Universal
  //from.push_back(bElBarLabel);
  //to.push_back(bElBarLabel_preReloc);
}

UCNH::~UCNH()
{
  // Plasticity Deletes
  if(d_usePlasticity)
  {
    VarLabel::destroy(p_statedata_label);
    VarLabel::destroy(p_statedata_label_preReloc);
  }
  
  // Universal Deletes
  VarLabel::destroy(bElBarLabel);
  VarLabel::destroy(bElBarLabel_preReloc);
}

void UCNH::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","UCNH");
  }
  
  cm_ps->appendElement("bulk_modulus",d_initialData.Bulk);
  cm_ps->appendElement("shear_modulus",d_initialData.Shear);
  cm_ps->appendElement("useModifiedEOS",d_useModifiedEOS);
  cm_ps->appendElement("usePlasticity", d_usePlasticity);
  cm_ps->appendElement("useDamage", d_useDamage);
  // Plasticity
  if(d_usePlasticity)
  {
    cm_ps->appendElement("yield_stress",d_initialData.FlowStress);
    cm_ps->appendElement("hardening_modulus",d_initialData.K);
    cm_ps->appendElement("alpha",d_initialData.Alpha);
  }
  
  // Damage
  if(d_useDamage)
  {
    
  }
}

UCNH* UCNH::clone()
{
  return scinew UCNH(*this);
}

void UCNH::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 Identity, zero(0.);
  Identity.Identity();
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  
  // Plasticity
  if(d_usePlasticity)
  {
    ParticleVariable<StateData> statedata;

    new_dw->allocateAndPut(statedata, p_statedata_label,  pset);
    
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end(); iter++){
      statedata[*iter].Alpha = d_initialData.Alpha;
    }
    ParticleVariable<Matrix3> bElBar;
  
    new_dw->allocateAndPut(bElBar,    bElBarLabel,        pset);
    iter = pset->begin();
    for(;iter != pset->end(); iter++){
      bElBar[*iter] = Identity;
    }
  }
  
  // Damage
  if(d_useDamage)
  {
    
  }
  
  // Universal
  /*P7YarticleVariable<Matrix3> bElBar;
  
  new_dw->allocateAndPut(bElBar,    bElBarLabel,        pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++){
    bElBar[*iter] = Identity;
  }*/
  
  // End by computing the stable timestep
  computeStableTimestep(patch, matl, new_dw);
}

void UCNH::allocateCMDataAddRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches,
                                            MPMLabel* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);
  
  // Add requires local to this model
  // Plasticity
  if(d_usePlasticity)
  {
    task->requires(Task::NewDW,p_statedata_label_preReloc, 
                   matlset, Ghost::None);
    task->requires(Task::NewDW,bElBarLabel_preReloc, 
                   matlset, Ghost::None);
  }

  // Universal
 // task->requires(Task::NewDW,bElBarLabel_preReloc, 
  //               matlset, Ghost::None);
}


void UCNH::allocateCMDataAdd(DataWarehouse* new_dw,
                                    ParticleSubset* addset,
                                    map<const VarLabel*, ParticleVariableBase*>* newState,
                                    ParticleSubset* delset,
                                    DataWarehouse* )
{
  // Copy the data common to all constitutive models from the particle to be 
  // deleted to the particle to be added. 
  // This method is defined in the ConstitutiveModel base class.
  copyDelToAddSetForConvertExplicit(new_dw, delset, addset, newState);
  
  // Copy the data local to this constitutive model from the particles to 
  // be deleted to the particles to be added
  
  // Plasticity
  if(d_usePlasticity)
  {
    ParticleVariable<StateData> statedata;
    constParticleVariable<StateData> o_statedata;
  
    new_dw->allocateTemporary(statedata,addset);
  
    new_dw->get(o_statedata,p_statedata_label_preReloc,delset);
  
    ParticleSubset::iterator o,n = addset->begin();
    for (o=delset->begin(); o != delset->end(); o++, n++) {
      statedata[*n].Alpha = o_statedata[*o].Alpha;
    }
  
    (*newState)[p_statedata_label]=statedata.clone();
  
  ParticleVariable<Matrix3>   bElBar;
  constParticleVariable<Matrix3>   o_bElBar;
  
  new_dw->allocateTemporary(bElBar,addset);
  
  new_dw->get(o_bElBar,bElBarLabel_preReloc,delset);
  
  ParticleSubset::iterator e = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, e++) {
    bElBar[*e] = o_bElBar[*o];
  }
  
  (*newState)[ bElBarLabel]=bElBar.clone();
  } // End Plasticity
  
  // Damage
  if(d_useDamage)
  {
    
  } // End Damage
  
  // Universal
/*  ParticleVariable<Matrix3>   bElBar;
  constParticleVariable<Matrix3>   o_bElBar;
  
  new_dw->allocateTemporary(bElBar,addset);
  
  new_dw->get(o_bElBar,bElBarLabel_preReloc,delset);
  
  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    bElBar[*n] = o_bElBar[*o];
  }
  
  (*newState)[ bElBarLabel]=bElBar.clone();
*/
}

void UCNH::computeStableTimestep(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel, pset);
  new_dw->get(pvolume,   lb->pVolumeLabel, pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double mu = d_initialData.Shear;
  double bulk = d_initialData.Bulk;
  
  if(d_usePlasticity)
  {
    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;
      
      // Compute wave speed at each particle, store the maximum
      Vector pvelocity_idx = pvelocity[idx];
      if(pmass[idx] > 0){
        c_dil = sqrt((bulk + 4.*mu/3.)*pvolume[idx]/pmass[idx]);
      }
      else{
        c_dil = 0.0;
        pvelocity_idx = Vector(0.0,0.0,0.0);
      }
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }
  } else {
    for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
      particleIndex idx = *iter;

      // Compute wave speed at each particle, store the maximum
      c_dil = sqrt((bulk + 4.*mu/3.)*pvolume[idx]/pmass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                        Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                        Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void UCNH::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
/*  Matrix3 Shear,bEB_new,deformationGradientInc, shearTrial,normal,fbar, bElBarTrial;
  double J,p,IEl,U,W,se=0.;
  double c_dil=0.0;
  double muBar, delgamma, fTrial, sTnorm, Jinc;
  double onethird = (1.0/3.0), sqtwthds = sqrt(2.0/3.0);
  Matrix3 Identity;
  Identity.Identity();
  double shear = d_initialData.Shear;
  double bulk  = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();
*/

  for(int pp=0;pp<patches->size();pp++){
  Matrix3 Shear,bEB_new,deformationGradientInc, shearTrial,normal,fbar, bElBarTrial;
  double J,p,IEl,U,W,se=0.;
  double c_dil=0.0;
  double muBar, delgamma, fTrial, sTnorm, Jinc;
  double onethird = (1.0/3.0), sqtwthds = sqrt(2.0/3.0);
  Matrix3 Identity;
  Identity.Identity();
  double shear = d_initialData.Shear;
  double bulk  = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();


    const Patch* patch = patches->get(pp);

    Vector dx = patch->dCell();

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    Ghost::GhostType  gac   = Ghost::AroundCells;

    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new, bElBar_new;
    constParticleVariable<Matrix3> deformationGradient, bElBar;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<StateData> statedata_old;
    ParticleVariable<StateData> statedata;
    constParticleVariable<double> pmass;
    ParticleVariable<double> pvolume_new;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Vector> psize;
    ParticleVariable<double> pdTdt,p_q;
    constNCVariable<Vector> gvelocity;

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    //old_dw->get(bElBar,              bElBarLabel,                  pset);
    
    
    double flow;
    double K;
    // Plasticity gets
    if(d_usePlasticity)
    {
      old_dw->get(bElBar,              bElBarLabel,                  pset);
      old_dw->get(statedata_old,         p_statedata_label,                pset);
      new_dw->allocateAndPut(statedata,  p_statedata_label_preReloc,       pset);
      new_dw->allocateAndPut(bElBar_new,  bElBarLabel_preReloc,      pset);
      
      statedata.copyData(statedata_old);
      
      flow  = d_initialData.FlowStress;
      K     = d_initialData.K;
    }
    
    //new_dw->allocateAndPut(bElBar_new,  bElBarLabel_preReloc,      pset);
    new_dw->allocateAndPut(pstress,     lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pvolume_new, lb->pVolumeLabel_preReloc, pset);
    new_dw->allocateAndPut(pdTdt,       lb->pdTdtLabel_preReloc,   pset);
    new_dw->allocateAndPut(deformationGradient_new,
                            lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(p_q,   lb->p_qLabel_preReloc,           pset);
    ParticleVariable<Matrix3> velGrad;
    new_dw->allocateTemporary(velGrad, pset);


    if(flag->d_doGridReset){
      constNCVariable<Vector> gvelocity;
      new_dw->get(gvelocity, lb->gVelocityStarLabel,dwi,patch,gac,NGN);

      double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
      for(ParticleSubset::iterator iter=pset->begin();iter!=pset->end();iter++){        particleIndex idx = *iter;

        Matrix3 tensorL(0.0);
        if(!flag->d_axisymmetric){
         // Get the node indices that surround the cell
         interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

         computeVelocityGradient(tensorL,ni,d_S, oodx, gvelocity);
        } else {  // axi-symmetric kinematics
         // Get the node indices that surround the cell
         interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                                    psize[idx],deformationGradient[idx]);
         // x -> r, y -> z, z -> theta
         computeAxiSymVelocityGradient(tensorL,ni,d_S,S,oodx,gvelocity,px[idx]);
        }
        velGrad[idx]=tensorL;

        deformationGradient_new[idx]=(tensorL*delT+Identity)
                                    *deformationGradient[idx];
      }
    }
    else if(!flag->d_doGridReset){
      constNCVariable<Vector> gdisplacement;
      new_dw->get(gdisplacement, lb->gDisplacementLabel,dwi,patch,gac,NGN);
      computeDeformationGradientFromDisplacement(gdisplacement,
                                                 pset, px, psize,
                                                 deformationGradient_new,
                                                 deformationGradient,
                                                 dx, interpolator);
    }

    // The following is used only for pressure stabilization
    CCVariable<double> J_CC;
    new_dw->allocateTemporary(J_CC,     patch);
    J_CC.initialize(0.);
    if(flag->d_doPressureStabilization) {
      CCVariable<double> vol_0_CC;
      CCVariable<double> vol_CC;
      new_dw->allocateTemporary(vol_0_CC, patch);
      new_dw->allocateTemporary(vol_CC, patch);

      vol_0_CC.initialize(0.);
      vol_CC.initialize(0.);
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
  
        // get the volumetric part of the deformation
        J = deformationGradient_new[idx].Determinant();
  
        // Get the deformed volume
        pvolume_new[idx]=(pmass[idx]/rho_orig)*J;
  
        IntVector cell_index;
        patch->findCell(px[idx],cell_index);
  
        vol_CC[cell_index]+=pvolume_new[idx];
        vol_0_CC[cell_index]+=pmass[idx]/rho_orig;
      }

      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        J_CC[c]=vol_CC[c]/vol_0_CC[c];
      }
    }

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      if(flag->d_doPressureStabilization) {
        IntVector cell_index;
        patch->findCell(px[idx],cell_index);

        // get the original volumetric part of the deformation
        J = deformationGradient_new[idx].Determinant();

        // Change F such that the determinant is equal to the average for
        // the cell
        deformationGradient_new[idx]*=cbrt(J_CC[cell_index])/cbrt(J);
      }

      // Plasticity Diff 1
      double alpha;
      if(d_usePlasticity)
      {
        // Calculate the stress Tensor (symmetric 3 x 3 Matrix) given the
        // time step and the velocity gradient and the material constants
        alpha = statedata[idx].Alpha;
        
        // Compute the deformation gradient increment
        deformationGradientInc = deformationGradient_new[idx]
        *deformationGradient[idx].Inverse();
        
        // get the volumetric part of the deformation
        Jinc = deformationGradientInc.Determinant();
      }
      
      J = deformationGradient_new[idx].Determinant();

      // Plasticity Diff 2
      if(d_usePlasticity)
      {
        // Check 1: Look at Jacobian
        if (!(J > 0.0)) {
          cerr << getpid() ;
          constParticleVariable<long64> pParticleID;
          old_dw->get(pParticleID, lb->pParticleIDLabel, pset);
          cerr << "**ERROR** Negative Jacobian of deformation gradient"
          << " in particle " << pParticleID[idx] << endl;
          cerr << "F_old = " << deformationGradient[idx] << endl;
          cerr << "F_inc = " << deformationGradientInc << endl;
          cerr << "F_new = " << deformationGradient_new[idx] << endl;
          cerr << "J = " << J << endl;
          //        deformationGradient_new[idx]=Identity;
          //        deformationGradientInc=Identity;
          //        J=1.;
          //        Jinc=1.;
        }
        
        // get the volume preserving part of the deformation gradient increment
        fbar = deformationGradientInc/cbrt(Jinc);
        
        // predict the elastic part of the volume preserving part of the left
        // Cauchy-Green deformation tensor
        bElBarTrial = fbar*bElBar[idx]*fbar.Transpose();
        IEl = onethird*bElBarTrial.Trace();
        
        // shearTrial is equal to the shear modulus times dev(bElBar)
        shearTrial = (bElBarTrial - Identity*IEl)*shear;
      }
      
      // Get the deformed volume
      //pvolume_new[idx]=(pmass[idx]/rho_orig)*J;

      // Compute local wave speed
      double rho_cur = rho_orig/J;
      pvolume_new[idx]=(pmass[idx]/rho_cur);
      c_dil = sqrt((bulk + 4.*shear/3.)/rho_cur);

      // Plasticity Diff 3
      if(d_usePlasticity)
      {
        // get the hydrostatic part of the stress
        p = 0.5*bulk*(J - 1.0/J);
        
        // Compute ||shearTrial||
        sTnorm = shearTrial.Norm();
        
        muBar = IEl * shear;
        
        // Check for plastic loading
        fTrial = sTnorm - sqtwthds*(K*alpha + flow);
        
        if(fTrial > 0.0){
          // plastic
          
          delgamma = (fTrial/(2.0*muBar)) / (1.0 + (K/(3.0*muBar)));
          
          normal = shearTrial/sTnorm;
          
          // The actual elastic shear stress
          Shear = shearTrial - normal*2.0*muBar*delgamma;
          
          // Deal with history variables
          statedata[idx].Alpha = alpha + sqtwthds*delgamma;
          bElBar_new[idx] = Shear/shear + Identity*IEl;
        }
        else {
          // not plastic
          
          bElBar_new[idx] = bElBarTrial;
          Shear = shearTrial;
        }        
      } else { // Non plasticity stuff
        double cubeRootJ=cbrt(J);
        double Jtothetwothirds=cubeRootJ*cubeRootJ;
        bEB_new = deformationGradient_new[idx]
                     * deformationGradient_new[idx].Transpose()/Jtothetwothirds;
        bElBar_new[idx] = bEB_new;
        IEl = onethird*bEB_new.Trace();

        // Shear is equal to the shear modulus times dev(bElBar)
        Shear = (bEB_new - Identity*IEl)*shear;

        // get the hydrostatic part of the stress
        p = 0.5*bulk*(J - 1.0/J);
      }
      
      // compute the total stress (volumetric + deviatoric)
      pstress[idx] = Identity*p + Shear/J;

      // Compute the strain energy for all the particles
      U = .5*bulk*(.5*(J*J - 1.0) - log(J));
      W = .5*shear*(bEB_new.Trace() - 3.0);
      double e = (U + W)*pvolume_new[idx]/J;
      se += e;

      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(bulk/rho_cur);
        Matrix3 D=(velGrad[idx] + velGrad[idx].Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);

    delete interpolator;
  }
}

void UCNH::carryForward(const PatchSubset* patches,
                               const MPMMaterial* matl,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Carry forward the data common to all constitutive models 
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);

    // Carry forward the data local to this constitutive model 
    
    // Universal
    ParticleVariable<Matrix3> bElBar_new;
    constParticleVariable<Matrix3> bElBar;
    old_dw->get(bElBar,                bElBarLabel,                    pset);
    new_dw->allocateAndPut(bElBar_new, bElBarLabel_preReloc,           pset);
    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;
      bElBar_new[idx] = bElBar[idx];
    }
    
    // Plasticity
    if(d_usePlasticity)
    {
      ParticleVariable<StateData> statedata;
      constParticleVariable<StateData> statedata_old;
      old_dw->get(statedata_old,         p_statedata_label,              pset);
      new_dw->allocateAndPut(statedata,  p_statedata_label_preReloc,     pset);
      statedata.copyData(statedata_old);
    }
    
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void UCNH::addInitialComputesAndRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  // Plasticity
  if(d_usePlasticity)
  {
    task->computes(p_statedata_label, matlset);
  task->computes(bElBarLabel,       matlset);
  
  }
  // Universal
  //task->computes(bElBarLabel,       matlset);
}

void UCNH::addComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);
  
  // Other constitutive model and input dependent computes and requires
  Ghost::GhostType  gnone = Ghost::None;
  
  // Plasticity
  if(d_usePlasticity)
  {
    task->requires(Task::OldDW, p_statedata_label, matlset,gnone);
    task->computes(p_statedata_label_preReloc, matlset);
    task->requires(Task::OldDW, bElBarLabel,       matlset,gnone);
    task->computes(bElBarLabel_preReloc,       matlset);
  }
  
  // Universal
  //task->requires(Task::OldDW, bElBarLabel,       matlset,gnone);
  //task->computes(bElBarLabel_preReloc,       matlset);
}

void 
UCNH::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool,
                                   const bool ) const
{
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double UCNH::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;
  
  double p_gauge = pressure - p_ref;
  double rho_cur;
 
  if(d_useModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;           // MODIFIED EOS
    double n = p_ref/bulk;
    rho_cur = rho_orig*pow(pressure/A,n);
  } else {                      // STANDARD EOS
    double p_g_over_bulk = p_gauge/bulk;
    rho_cur=rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));
  }
  return rho_cur;
}

void UCNH::computePressEOSCM(const double rho_cur,double& pressure, 
                                    const double p_ref,
                                    double& dp_drho, double& tmp,
                                    const MPMMaterial* matl)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  if(d_useModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;           // MODIFIED EOS
    double n = bulk/p_ref;
    double rho_rat_to_the_n = pow(rho_cur/rho_orig,n);
    pressure = A*rho_rat_to_the_n;
    dp_drho  = (bulk/rho_cur)*rho_rat_to_the_n;
    tmp      = dp_drho;         // speed of sound squared
  } else {                      // STANDARD EOS            
    double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
    tmp        = bulk/rho_cur;  // speed of sound squared
  }
}

double UCNH::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}



// Damage Specific Functions

void UCNH::addRequiresDamageParameter(Uintah::Task*, Uintah::MPMMaterial const*, Uintah::ComputeSet<Uintah::Patch const*> const*) const {
}

void UCNH::getDamageParameter(Uintah::Patch const*, Uintah::ParticleVariable<int>&, int, Uintah::DataWarehouse*, Uintah::DataWarehouse*) {
}

void UCNH::computeStressTensor(Uintah::ComputeSubset<Uintah::Patch const*> const*, Uintah::MPMMaterial const*, Uintah::DataWarehouse*, Uintah::DataWarehouse*, Uintah::Solver*, bool) {
}


namespace Uintah {
   
static MPI_Datatype makeMPI_CMData()
  {
    ASSERTEQ(sizeof(UCNH::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 1, 1, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(UCNH::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "UCNH::StateData", 
                                  true, &makeMPI_CMData);
    }
    return td;
  }
} // End namespace Uintah
