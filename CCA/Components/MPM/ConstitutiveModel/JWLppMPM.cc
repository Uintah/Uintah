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


#include <CCA/Components/MPM/ConstitutiveModel/JWLppMPM.h>
#include <Core/Malloc/Allocator.h>
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
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

JWLppMPM::JWLppMPM(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  d_useModifiedEOS = false;

  // Read the ignition pressure
  ps->require("ignition_pressure", d_initialData.ignition_pressure);

  // These two parameters are used for the unburned Murnahan EOS
  ps->require("murnaghan_K",    d_initialData.K);
  ps->require("murnaghan_n",    d_initialData.n);

  // These parameters are used for the product JWL EOS
  ps->require("jwl_A",    d_initialData.A);
  ps->require("jwl_B",    d_initialData.B);
  ps->require("jwl_C",    d_initialData.C);
  ps->require("jwl_R1",   d_initialData.R1);
  ps->require("jwl_R2",   d_initialData.R2);
  ps->require("jwl_om",   d_initialData.omega);
  ps->require("jwl_rho0", d_initialData.rho0);

  // These parameters are needed for the reaction model
  ps->require("reaction_G",    d_initialData.G); // Rate coefficient
  ps->require("reaction_b",    d_initialData.b); // Pressure exponent

  // Initial stress
  // Fix: Need to make it more general.  Add gravity turn-on option and 
  //      read from file option etc.
  ps->getWithDefault("useInitialStress", d_useInitialStress, false);
  d_init_pressure = 0.0;
  if (d_useInitialStress) {
    ps->getWithDefault("initial_pressure", d_init_pressure, 0.0);
  } 

  pProgressFLabel             = VarLabel::create("p.progressF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressFLabel_preReloc    = VarLabel::create("p.progressF+",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel          = VarLabel::create("p.progressdelF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel_preReloc = VarLabel::create("p.progressdelF+",
                               ParticleVariable<double>::getTypeDescription());
  pVelGradLabel               = VarLabel::create("p.velGrad",
                               ParticleVariable<double>::getTypeDescription());
  pVelGradLabel_preReloc      = VarLabel::create("p.velGrad+",
                               ParticleVariable<double>::getTypeDescription());
}

JWLppMPM::JWLppMPM(const JWLppMPM* cm) : ConstitutiveModel(cm)
{
  d_useModifiedEOS = cm->d_useModifiedEOS ;

  d_initialData.ignition_pressure = cm->d_initialData.ignition_pressure;

  d_initialData.K = cm->d_initialData.K;
  d_initialData.n = cm->d_initialData.n;

  d_initialData.A = cm->d_initialData.A;
  d_initialData.B = cm->d_initialData.B;
  d_initialData.C = cm->d_initialData.C;
  d_initialData.R1 = cm->d_initialData.R1;
  d_initialData.R2 = cm->d_initialData.R2;
  d_initialData.omega = cm->d_initialData.omega;
  d_initialData.rho0 = cm->d_initialData.rho0;

  d_initialData.G    = cm->d_initialData.G;
  d_initialData.b    = cm->d_initialData.b;

  // Initial stress
  d_useInitialStress = cm->d_useInitialStress;
  d_init_pressure = cm->d_init_pressure;

  pProgressFLabel          = VarLabel::create("p.progressF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressFLabel_preReloc = VarLabel::create("p.progressF+",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel          = VarLabel::create("p.progressdelF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel_preReloc = VarLabel::create("p.progressdelF+",
                               ParticleVariable<double>::getTypeDescription());
  pVelGradLabel               = VarLabel::create("p.velGrad",
                               ParticleVariable<double>::getTypeDescription());
  pVelGradLabel_preReloc      = VarLabel::create("p.velGrad+",
                               ParticleVariable<double>::getTypeDescription());
}

JWLppMPM::~JWLppMPM()
{
  VarLabel::destroy(pProgressFLabel);
  VarLabel::destroy(pProgressFLabel_preReloc);
  VarLabel::destroy(pProgressdelFLabel);
  VarLabel::destroy(pProgressdelFLabel_preReloc);
  VarLabel::destroy(pVelGradLabel);
  VarLabel::destroy(pVelGradLabel_preReloc);
}

void JWLppMPM::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","jwlpp_mpm");
  }
  
  cm_ps->appendElement("ignition_pressure", d_initialData.ignition_pressure);

  cm_ps->appendElement("murnaghan_K", d_initialData.K);
  cm_ps->appendElement("murnaghan_n", d_initialData.n);

  cm_ps->appendElement("jwl_A",    d_initialData.A);
  cm_ps->appendElement("jwl_B",    d_initialData.B);
  cm_ps->appendElement("jwl_C",    d_initialData.C);
  cm_ps->appendElement("jwl_R1",   d_initialData.R1);
  cm_ps->appendElement("jwl_R2",   d_initialData.R2);
  cm_ps->appendElement("jwl_om",   d_initialData.omega);
  cm_ps->appendElement("jwl_rho0", d_initialData.rho0);

  cm_ps->appendElement("reaction_b", d_initialData.b);
  cm_ps->appendElement("reaction_G", d_initialData.G);

  cm_ps->appendElement("useInitialStress", d_useInitialStress);
  if (d_useInitialStress) {
    cm_ps->appendElement("initial_pressure", d_init_pressure);
  }
}

JWLppMPM* JWLppMPM::clone()
{
  return scinew JWLppMPM(*this);
}

void JWLppMPM::initializeCMData(const Patch* patch,
                             const MPMMaterial* matl,
                             DataWarehouse* new_dw)
{
  // Initialize local variables
  Matrix3 zero(0.0);
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<double> pProgress, pProgressdelF;
  ParticleVariable<Matrix3> pVelGrad;
  new_dw->allocateAndPut(pProgress,pProgressFLabel,pset);
  new_dw->allocateAndPut(pProgressdelF,pProgressdelFLabel,pset);
  new_dw->allocateAndPut(pVelGrad, pVelGradLabel, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    pProgress[*iter]     = 0.0;
    pProgressdelF[*iter] = 0.0;
    pVelGrad[*iter] = zero;
  }

  // Initialize the variables shared by all constitutive models
  if (!d_useInitialStress) {
    // This method is defined in the ConstitutiveModel base class.
    initSharedDataForExplicit(patch, matl, new_dw);

  } else {
    // Initial stress option 
    Matrix3 Identity;
    Identity.Identity();
    Matrix3 zero(0.0);

    ParticleVariable<double>  pdTdt;
    ParticleVariable<Matrix3> pDefGrad;
    ParticleVariable<Matrix3> pStress;

    new_dw->allocateAndPut(pdTdt,       lb->pdTdtLabel,               pset);
    new_dw->allocateAndPut(pDefGrad,    lb->pDeformationMeasureLabel, pset);
    new_dw->allocateAndPut(pStress,     lb->pStressLabel,             pset);

    // Set the initial pressure
    double p = d_init_pressure;
    Matrix3 sigInit(-p, 0.0, 0.0, 0.0, -p, 0.0, 0.0, 0.0, -p);

    // Compute deformation gradient
    //  using the Murnaghan eos 
    //     p = (1/nK) [J^(-n) - 1]
    //     =>
    //     det(F) = (1 + nKp)^(-1/n)
    //     =>
    //     F_{11} = F_{22} = F_{33} = (1 + nKp)^(-1/3n)
    double F11 = pow((1.0 + d_initialData.K*d_initialData.n*p), (-1.0/(3.0*d_initialData.n)));
    Matrix3 defGrad(F11, 0.0, 0.0, 0.0, F11, 0.0, 0.0, 0.0, F11);

    iter = pset->begin();
    for(;iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pdTdt[idx] = 0.0;
      pStress[idx] = sigInit;
      pDefGrad[idx] = defGrad;
    }
  }

  computeStableTimestep(patch, matl, new_dw);
}

void JWLppMPM::allocateCMDataAddRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches,
                                            MPMLabel* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);
}


void JWLppMPM::allocateCMDataAdd(DataWarehouse* new_dw,
                                    ParticleSubset* addset,
                                    map<const VarLabel*,
                                    ParticleVariableBase*>* newState,
                                    ParticleSubset* delset,
                                    DataWarehouse* )
{
  // Copy the data common to all constitutive models from the particle to be 
  // deleted to the particle to be added. 
  // This method is defined in the ConstitutiveModel base class.
  copyDelToAddSetForConvertExplicit(new_dw, delset, addset, newState);
  
  // Copy the data local to this constitutive model from the particles to 
  // be deleted to the particles to be added
}

void JWLppMPM::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pProgressFLabel);
  to.push_back(pProgressFLabel_preReloc);
  from.push_back(pProgressdelFLabel);
  to.push_back(pProgressdelFLabel_preReloc);
  from.push_back(pVelGradLabel);
  to.push_back(pVelGradLabel_preReloc);
}

void JWLppMPM::computeStableTimestep(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume,ptemperature;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,        lb->pMassLabel,        pset);
  new_dw->get(pvolume,      lb->pVolumeLabel,      pset);
  new_dw->get(pvelocity,    lb->pVelocityLabel,    pset);
  new_dw->get(ptemperature, lb->pTemperatureLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double K    = d_initialData.K;
  double n    = d_initialData.n;
  double rho0 = d_initialData.rho0;
  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
     particleIndex idx = *iter;
     // Compute wave speed at each particle, store the maximum
     double rhoM = pmass[idx]/pvolume[idx];
     double dp_drho = (1./(K*rho0))*pow((rhoM/rho0),n-1.);
     c_dil = sqrt(dp_drho);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void JWLppMPM::computeStressTensor(const PatchSubset* patches,
                                   const MPMMaterial* matl,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  // Constants 
  Vector WaveSpeed(1.e-12, 1.e-12, 1.e-12);
  Matrix3 Identity;
  Identity.Identity();

  // Material parameters
  double d_ignition_pressure = d_initialData.ignition_pressure;
  double d_K = d_initialData.K;
  double d_n = d_initialData.n;
  double d_A = d_initialData.A;
  double d_B = d_initialData.B;
  double d_C = d_initialData.C;
  double d_R1 = d_initialData.R1;
  double d_R2 = d_initialData.R2;
  double d_omega = d_initialData.omega;
  double d_rho0 = d_initialData.rho0; // matl->getInitialDensity();

  double d_b = d_initialData.b;
  double d_G = d_initialData.G;

  // Loop through patches
  for(int pp=0; pp<patches->size(); pp++){
    const Patch* patch = patches->get(pp);

    double se  = 0.0;
    double c_dil = 0.0;

    // Get data warehouse, particle set, and patch info
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    // double time = d_sharedState->getElapsedTime();

    // Get interpolator
    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector>    d_S(interpolator->size());
    vector<double>    S(interpolator->size());

    // variables to hold this timestep's values
    constParticleVariable<double>  pmass, pProgressF, pProgressdelF, pvolume_old;
    ParticleVariable<double>       pvolume;
    ParticleVariable<double>       pdTdt, p_q, pProgressF_new, pProgressdelF_new;
    constParticleVariable<Vector>  pvelocity;
    constParticleVariable<Matrix3> psize;
    constParticleVariable<Point>   px;
    constParticleVariable<Matrix3> pDefGrad, pstress;
    constParticleVariable<Matrix3> pVelGrad;
    ParticleVariable<Matrix3>      pVelGrad_new;
    ParticleVariable<Matrix3>      pDefGrad_new;
    ParticleVariable<Matrix3>      pstress_new;


    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    Ghost::GhostType  gac   = Ghost::AroundCells;
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(pDefGrad, lb->pDeformationMeasureLabel, pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    old_dw->get(pvolume_old,         lb->pVolumeLabel,             pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pProgressF,          pProgressFLabel,              pset);
    old_dw->get(pProgressdelF,       pProgressdelFLabel,           pset);
    old_dw->get(pVelGrad,            pVelGradLabel,                pset);
    
    new_dw->allocateAndPut(pstress_new,      lb->pStressLabel_preReloc,    pset);
    new_dw->allocateAndPut(pvolume,          lb->pVolumeLabel_preReloc,    pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,      pset);
    new_dw->allocateAndPut(p_q,              lb->p_qLabel_preReloc,        pset);
    new_dw->allocateAndPut(pDefGrad_new,
                                  lb->pDeformationMeasureLabel_preReloc,   pset);

    new_dw->allocateAndPut(pProgressF_new,    pProgressFLabel_preReloc,    pset);
    new_dw->allocateAndPut(pProgressdelF_new, pProgressdelFLabel_preReloc, pset);

    new_dw->allocateAndPut(pVelGrad_new,      pVelGradLabel_preReloc,      pset);

    constNCVariable<Vector> gvelocity;
    new_dw->get(gvelocity, lb->gVelocityStarLabel, dwi, patch, gac, NGN);


    if(!flag->d_doGridReset){
      cerr << "The jwlpp_mpm model doesn't work without resetting the grid"
           << endl;
    }

    // Compute deformation gradient and velocity gradient at each 
    // particle before pressure stabilization
    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      Matrix3 velGrad_new(0.0);
      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
                                                  pDefGrad[idx]);

        // standard computation
        //cerr << "JWL++::computeVelocityGradient for particle: " << idx  << " patch = " << pp 
        //     << " px = " << px[idx] <<  " pmass = " << pmass[idx] << " pvelocity = " << pvelocity[idx] << endl;
        computeVelocityGradient(velGrad_new, ni, d_S, oodx, gvelocity);
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx],
                                                            pDefGrad[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad_new,ni,d_S,S,oodx,gvelocity,
                                                                  px[idx]);
      }

      pVelGrad_new[idx] = velGrad_new;
      //if (isnan(velGrad_new.Norm())) {
      //  cerr << "particle = " << idx << " velGrad = " << velGrad_new << endl;
      //  throw InvalidValue("**ERROR**: Nan in velocity gradient value", __FILE__, __LINE__);
      //}
      //pDefGrad_new[idx]=(velGrad_new*delT+Identity)*pDefGrad[idx];

      // Improve upon first order estimate of deformation gradient
      // Compute mid point velocity gradient
      // Matrix3 Amat = (pVelGrad[idx] + pVelGrad_new[idx])*(0.5*delT);
      // int num_terms = 100;
      // Matrix3 Finc = Amat.Exponential(num_terms);
      // pDefGrad_new[idx] = Finc*pDefGrad[idx];
      // Matrix3 Finc_old = velGrad_new*delT + Identity;
      // Matrix3 Finc_diff = Finc_old - Finc;
      // if (Finc_diff.Norm() > 0.1*Finc_old.Norm()) {
      //    cerr << "Huge diff in F calcs." << " Finc_old = " << Finc_old << " Finc_new = " << Finc << " Amat = " << Amat << endl;
      // } 

      // Improve upon first order estimate of deformation gradient
      Matrix3 F = pDefGrad[idx];
      double Lnorm_dt = velGrad_new.Norm()*delT;
      int num_subcycles = max(1,2*((int) Lnorm_dt));
      if(num_subcycles > 1000) {
         cout << "NUM_SCS = " << num_subcycles << endl;
      }
      double dtsc = delT/(double (num_subcycles));
      Matrix3 OP_tensorL_DT = Identity + velGrad_new*dtsc;
      for(int n=0; n<num_subcycles; n++){
         F = OP_tensorL_DT*F;
      }
      pDefGrad_new[idx] = F;
    }

    // The following is used only for pressure stabilization
    CCVariable<double> J_CC;
    new_dw->allocateTemporary(J_CC,       patch);
    J_CC.initialize(0.);
    if(flag->d_doPressureStabilization) {
      CCVariable<double> vol_0_CC;
      CCVariable<double> vol_CC;
      new_dw->allocateTemporary(vol_0_CC, patch);
      new_dw->allocateTemporary(vol_CC,   patch);

      vol_0_CC.initialize(0.);
      vol_CC.initialize(0.);
      iter = pset->begin();
      for(; iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // get the volumetric part of the deformation
        double J = pDefGrad_new[idx].Determinant();

        // Get the deformed volume
        double rho_cur = d_rho0/J;
        pvolume[idx] = pmass[idx]/rho_cur;

        IntVector cell_index;
        patch->findCell(px[idx],cell_index);

        vol_CC[cell_index]  +=pvolume[idx];
        vol_0_CC[cell_index]+=pmass[idx]/d_rho0;
      }

      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        J_CC[c]=vol_CC[c]/vol_0_CC[c];
      }
    } //end of pressureStabilization loop  at the patch level

    // Actually compute the updated stress 
    iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      double J = pDefGrad_new[idx].Determinant();
      if (!(J > 0.0)) {
        cerr << "**ERROR in JWL++MPM** Negative Jacobian of deformation gradient" << endl;
        cerr << "idx = " << idx << " J = " << J << " matl = " << matl << endl;
        cerr << "F_old = " << pDefGrad[idx]     << endl;
        cerr << "F_new = " << pDefGrad_new[idx] << endl;
        cerr << "VelGrad = " << pVelGrad_new[idx] << endl;
        throw InvalidValue("**ERROR**: Error in deformation gradient", __FILE__, __LINE__);
      }

      // More Pressure Stabilization
      if(flag->d_doPressureStabilization) {
        IntVector cell_index;
        patch->findCell(px[idx],cell_index);

        // Change F such that the determinant is equal to the average for
        // the cell
        pDefGrad_new[idx]*=cbrt(J_CC[cell_index])/cbrt(J);
        J=J_CC[cell_index];
      }

      // Compute new mass density and update the deformed volume
      double rho_cur = d_rho0/J;
      pvolume[idx] = pmass[idx]/rho_cur;

      // This is the burn logic used in the reaction model  (more complex versions
      //   are available -- see LS-DYNA manual)
      //       df/dt = G (1-f) p^b
      //       Forward Euler: f_{n+1} = f_n + G*(1-f_n)*p_n^b*delT
      //       Fourth-order R-K: f_{n+1} = f_n + 1/6(k1 + 2k2 + 2k3 + k4)
      //         k1 = G*(1-f_n)*p_n^b*delT
      //         k2 = G*(1-f_n-k1/2)*p_n^b*delT
      //         k3 = G*(1-f_n-k2/2)*p_n^b*delT
      //         k4 = G*(1-f_n-k3)*p_n^b*delT
      // (ignition_pressure in previous versions hardcoded to 2.0e8 Pa)
      double pressure = -(1.0/3.0)*pstress[idx].Trace();
      double f_old = pProgressF[idx];
      double f_inc = pProgressdelF[idx];
      double f_new = f_old;
      if(pressure > d_ignition_pressure)  
      {
        if (fabs(2.0*f_old - 1.0) <= 1.0) {
          int numCycles = (int) ceil(delT/5.0e-11);  // Time step harded at 5.0e-11 secs
          double delTinc = delT/((double)numCycles);
          for (int ii = 0; ii < numCycles; ++ii) {
            double fac = (delTinc*d_G)*pow(pressure, d_b);
            // Forward Euler
            // f_inc = (1.0 - f_old)*fac;
            // Fourth-order R-K
            double k1 = (1.0 - f_old)*fac;
            double k2 = (1.0 - f_old - 0.5*k1)*fac;
            double k3 = (1.0 - f_old - 0.5*k2)*fac;
            double k4 = (1.0 - f_old - k3)*fac;
            f_inc = 1.0/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4);
            f_new = f_old + f_inc;
          }
        }
        // if (fabs(2.0*f_new - 1.0) > 1.0) {
        //   cerr << " after cycling volume fraction burned is " << f_new << endl;
        // }
      }
      pProgressdelF_new[idx] = f_inc;
      pProgressF_new[idx] = f_new;

      //  The following computes a pressure for partially burned particles
      //  as a mixture of Murnaghan and JWL pressures, based on pProgressF
      //  This is as described in Eq. 5 of "JWL++: ..." by Souers, et al.
      double pM = (1.0/(d_n*d_K))*(pow(J,-d_n) - 1.0);
      double pJWL = pM;
      if(pProgressF_new[idx] > 0.0){
        double one_plus_omega = 1.0 + d_omega;
        double inv_rho_rat = J; //rho0/rhoM;
        double rho_rat = 1.0/J;  //rhoM/rho0;
        double A_e_to_the_R1_rho0_over_rhoM = d_A*exp(-d_R1*inv_rho_rat);
        double B_e_to_the_R2_rho0_over_rhoM = d_B*exp(-d_R2*inv_rho_rat);
        double C_rho_rat_tothe_one_plus_omega = d_C*pow(rho_rat, one_plus_omega);

        pJWL  = A_e_to_the_R1_rho0_over_rhoM +
                B_e_to_the_R2_rho0_over_rhoM +
                C_rho_rat_tothe_one_plus_omega;
      }

      double pressure_new = pM*(1.0 - pProgressF_new[idx]) + pJWL*pProgressF_new[idx];

      // compute the total stress
      pstress_new[idx] = Identity*(-pressure_new);
      if (isnan(pstress_new[idx].Norm())) {
        cerr << "particle = " << idx << " velGrad = " << pVelGrad_new[idx] << " stress_old = " << pstress[idx] << endl;
        cerr << " stress = " << pstress_new[idx] 
             << "  pProgressdelF_new = " << pProgressdelF_new[idx] 
             << "  pProgressF_new = " << pProgressF_new[idx] 
             << " pm = " << pM << " pJWL = " << pJWL <<  " rho_cur = " << rho_cur << endl;
        cerr << " pmass = " << pmass[idx] << " pvol = " << pvolume[idx] << endl;
        throw InvalidValue("**ERROR**: Nan in stress value", __FILE__, __LINE__);
      }

      Vector pvelocity_idx = pvelocity[idx];
      //if (isnan(pvelocity[idx].length())) {
      //  cerr << "particle = " << idx << " velocity = " << pvelocity[idx] << endl;
      //  throw InvalidValue("**ERROR**: Nan in particle velocity value", __FILE__, __LINE__);
      //}

      // Compute wave speed at each particle, store the maximum
      double dp_drho = (1./(d_K*d_rho0))*pow((rho_cur/d_rho0),d_n-1.);
      c_dil = sqrt(dp_drho);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
                                                                                
      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(1.0/(d_K*rho_cur));
        Matrix3 D=(pVelGrad_new[idx] + pVelGrad_new[idx].Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),      lb->StrainEnergyLabel);
    }
    delete interpolator;
  }
}

void JWLppMPM::carryForward(const PatchSubset* patches,
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
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),   lb->StrainEnergyLabel);  
    }
  }
}

void JWLppMPM::addComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  task->requires(Task::OldDW, pProgressFLabel,    matlset, Ghost::None);
  task->computes(pProgressFLabel_preReloc,        matlset);
  task->requires(Task::OldDW, pProgressdelFLabel, matlset, Ghost::None);
  task->computes(pProgressdelFLabel_preReloc,     matlset);
  task->requires(Task::OldDW, pVelGradLabel,      matlset, Ghost::None);
  task->computes(pVelGradLabel_preReloc,          matlset);
}

void JWLppMPM::addInitialComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet*) const
{ 
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pProgressFLabel,       matlset);
  task->computes(pProgressdelFLabel,    matlset);
  task->computes(pVelGradLabel,         matlset);
}

void 
JWLppMPM::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}


// This is not yet implemented - JG- 7/26/10
double JWLppMPM::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
    cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR JWLppMPM"
       << endl;
    double rho_orig = d_initialData.rho0; //matl->getInitialDensity();

    return rho_orig;
}

void JWLppMPM::computePressEOSCM(const double rhoM,double& pressure, 
                                    const double p_ref,
                                    double& dp_drho, double& tmp,
                                    const MPMMaterial* matl,
                                    double temperature)
{
  double A = d_initialData.A;
  double B = d_initialData.B;
  double R1 = d_initialData.R1;
  double R2 = d_initialData.R2;
  double omega = d_initialData.omega;
  double rho0 = d_initialData.rho0;
  double cv = matl->getSpecificHeat();
  double V = rho0/rhoM;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);
  double P3 = omega*cv*tmp*rhoM;

  pressure = P1 + P2 + P3;

  dp_drho = (R1*rho0*P1 + R2*rho0*P2)/(rhoM*rhoM) + omega*cv*tmp;
}

// This is not yet implemented - JG- 7/26/10
double JWLppMPM::getCompressibility()
{
   cout << "NO VERSION OF getCompressibility EXISTS YET FOR JWLppMPM"<< endl;
  return 1.0;
}

namespace Uintah {
} // End namespace Uintah
