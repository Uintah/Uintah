#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/Kayenta.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>

#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>

#include <sci_defs/uintah_defs.h>

#include <fstream>
#include <iostream>
#include <string>

////////////////////////////////////////////////////////////////////////////////
// The following functions are found in fortran/*.F

extern "C"{

#if defined( FORTRAN_UNDERSCORE_END )
#  define GEOCHK hookechk_
#  define ISOTROPIC_GEOMATERIAL_CALC isotropic_geomaterial_calc_
#elif defined( FORTRAN_UNDERSCORE_LINUX )
#  define GEOCHK hookechk_
#  define ISOTROPIC_GEOMATERIAL_CALC isotropic_geomaterial_calc__
#else // NONE
#  define GEOCHK hookechk
#  define ISOTROPIC_GEOMATERIAL_CALC isotropic_geommaterial_calc_
#endif

   void GEOCHK( double UI[], double UJ[], double UK[] );
   void ISOTROPIC_GEOMATERIAL_CALC( int &nblk, int &ninsv, double &dt,
                                    double UI[], double stress[], double D[],
                                    double svarg[], double &USM );
}

// End fortran functions.
////////////////////////////////////////////////////////////////////////////////

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

Kayenta::Kayenta(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  // Read model parameters from the input file
  getInputParameters(ps);

  // Check that model parameters are valid and allow model to change if needed
  GEOCHK(UI,UI,UI);

  //Create VarLabels for GeoModel internal state variables (ISVs)
  d_NINSV=37;
  initializeLocalMPMLabels();

}

Kayenta::Kayenta(const Kayenta* cm) : ConstitutiveModel(cm)
{
  for(int i=0;i<40;i++){
    UI[i] = cm->UI[i];
  }
}

Kayenta::~Kayenta()
{
}

void Kayenta::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","kayenta");
  }

  cm_ps->appendElement("B0",UI[0]);   // initial bulk modulus (stress)
  cm_ps->appendElement("B1",UI[1]);   // nonlinear bulk mod param (stress)
  cm_ps->appendElement("B2",UI[2]);   // nonlinear bulk mod param (stress)
  cm_ps->appendElement("B3",UI[3]);   // nonlinear bulk mod param (stress)
  cm_ps->appendElement("B4",UI[4]);   // nonlinear bulk mod param (dim-less)

  cm_ps->appendElement("G0",UI[5]);   // initial shear modulus (stress)
  cm_ps->appendElement("G1",UI[6]);   // nonlinear shear mod param
  cm_ps->appendElement("G2",UI[7]);   // nonlinear shear mod param (1/stres)
  cm_ps->appendElement("G3",UI[8]);   // nonlinear shear mod param (stress)
  cm_ps->appendElement("G4",UI[9]);   // nonlinear shear mod param 

  cm_ps->appendElement("RJS",UI[10]); // joint spacing (iso. joint set) 
  cm_ps->appendElement("RKS",UI[11]); // joint shear stiffness (iso. case)
  cm_ps->appendElement("RKN",UI[12]); // joint normal stiffness (iso. case) 

  cm_ps->appendElement("A1",UI[13]);  // meridional yld prof param (stress)
  cm_ps->appendElement("A2",UI[14]);  // meridional yld prof param (1/stres)
  cm_ps->appendElement("A3",UI[15]);  // meridional yld prof param (stress)
  cm_ps->appendElement("A4",UI[16]);  // meridional yld prof param

  cm_ps->appendElement("P0",UI[17]);  // init hydrostatic crush press 
  cm_ps->appendElement("P1",UI[18]);  // crush curve parameter (1/stress)
  cm_ps->appendElement("P2",UI[19]);  // crush curve parameter (1/stress^2)
  cm_ps->appendElement("P3",UI[20]);  // crush curve parameter (strain)

  cm_ps->appendElement("CR",UI[21]);  // cap curvature parameter (dim. less)
  cm_ps->appendElement("RK",UI[22]);  // TXE/TXC strength ratio (dim. less)
  cm_ps->appendElement("RN",UI[23]);  // TXE/TXC strength ratio (stress)
  cm_ps->appendElement("HC",UI[24]);  // kinematic hardening modulus (strs)

  cm_ps->appendElement("CTI1",UI[25]);// Tension I1 cut-off (stress)
  cm_ps->appendElement("CTPS",UI[26]);// Tension prin. stress cut-off (strs)

  cm_ps->appendElement("T1",UI[27]);  // rate dep. primary relax. time(time)
  cm_ps->appendElement("T2",UI[28]);  // rate dep. nonlinear param (1/time)
  cm_ps->appendElement("T3",UI[29]);  // rate dep. nonlinear param (dim-lss)
  cm_ps->appendElement("T4",UI[30]);  // not used (1/time)
  cm_ps->appendElement("T5",UI[31]);  // not used (stress)
  cm_ps->appendElement("T6",UI[32]);  // rate dep. nonlinear param (time)
  cm_ps->appendElement("T7",UI[33]);  // rate dep. nonlinear param (1/strs)

  cm_ps->appendElement("J3TYPE",UI[34]);// octahedral profile shape option
  cm_ps->appendElement("A2PF",UI[35]);  // flow potential analog of A2
  cm_ps->appendElement("A4PF",UI[36]);  // flow potential analog of A4
  cm_ps->appendElement("CRPF",UI[37]);  // flow potential analog of CR
  cm_ps->appendElement("RKPF",UI[38]);  // flow potential analog of RK
}

Kayenta* Kayenta::clone()
{
  return scinew Kayenta(*this);
}

void Kayenta::initializeCMData(const Patch* patch,
                               const MPMMaterial* matl,
                               DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  computeStableTimestep(patch, matl, new_dw);
}


void Kayenta::allocateCMDataAddRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);
}


void Kayenta::allocateCMDataAdd(DataWarehouse* new_dw,
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
}

void Kayenta::addParticleState(std::vector<const VarLabel*>& from,
                               std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  for(int i=1;i<=d_NINSV;i++){
    from.push_back(ISVLabels[i]);
    to.push_back(ISVLabels_preReloc[i]);
  }
}

void Kayenta::computeStableTimestep(const Patch* patch,
                                    const MPMMaterial* matl,
                                    DataWarehouse* new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double bulk = UI[0];
  double G = UI[5];
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk + 4.*G/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void Kayenta::computeStressTensor(const PatchSubset* patches,
                                  const MPMMaterial* matl,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  double rho_orig = matl->getInitialDensity();
  for(int p=0;p<patches->size();p++){
    double se = 0.0;
    const Patch* patch = patches->get(p);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    Matrix3 velGrad,deformationGradientInc,Identity,zero(0.),One(1.);
    double c_dil=0.0,Jinc;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
//    double onethird = (1.0/3.0);

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3> pstress_new;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<double> pmass, pvolume, ptemperature;
    ParticleVariable<double> pvolume_new;
    constParticleVariable<Vector> pvelocity, psize;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    Ghost::GhostType  gac   = Ghost::AroundCells;
    
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

    new_dw->get(gvelocity,lb->gVelocityStarLabel, dwi,patch, gac, NGN);

    ParticleVariable<double> pdTdt,p_q;

    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume_new,     lb->pVolumeLabel_preReloc,   pset);
    new_dw->allocateAndPut(pdTdt,           lb->pdTdtLabel_preReloc,     pset);
    new_dw->allocateAndPut(p_q,             lb->p_qLabel_preReloc,       pset);
    new_dw->allocateAndPut(deformationGradient_new,
                           lb->pDeformationMeasureLabel_preReloc,        pset);

    double UI[2];
    UI[0] = d_initialData.K;
    UI[1] = d_initialData.G;

    for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;
      // Initialize velocity gradient
      velGrad.set(0.0);

      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx]);

        computeVelocityGradient(velGrad,ni,d_S,oodx,gvelocity);

      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                                    psize[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad,ni,d_S,S,oodx,gvelocity,px[idx]);
      }

      // Calculate rate of deformation D, and deviatoric rate DPrime,
      Matrix3 D = (velGrad + velGrad.Transpose())*.5;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad * delT + Identity;

      Jinc = deformationGradientInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
                                     deformationGradient[idx];

      // get the volumetric part of the deformation
      double J = deformationGradient[idx].Determinant();
      pvolume_new[idx]=Jinc*pvolume[idx];

      // Compute the local sound speed
      double rho_cur = rho_orig/J;
       
      // This is the (updated) Cauchy stress
#if 0
      double onethird = 1./3.;
      Matrix3 DPrime = D - Identity*onethird*D.Trace();
      double G=UI[1];
      double bulk=UI[0];
      pstress_new[idx] = pstress[idx] + 
                         (DPrime*2.*G + Identity*bulk*D.Trace())*delT;

      cout << pstress_new[idx] << endl;
#endif

      double sigarg[6];
      sigarg[0]=pstress[idx](0,0);
      sigarg[1]=pstress[idx](1,1);
      sigarg[2]=pstress[idx](2,2);
      sigarg[3]=pstress[idx](0,1);
      sigarg[4]=pstress[idx](1,2);
      sigarg[5]=pstress[idx](2,0);
      double Darray[6];
      Darray[0]=D(0,0);
      Darray[1]=D(1,1);
      Darray[2]=D(2,2);
      Darray[3]=D(0,1);
      Darray[4]=D(1,2);
      Darray[5]=D(2,0);
      double svarg[1];
      double USM=9e99;
      double dt = delT;
      int nblk = 1;
      int ninsv = 1;

      ISOTROPIC_GEOMATERIAL_CALC(nblk,ninsv,dt, UI, sigarg, Darray, svarg, USM);

      pstress_new[idx](0,0) = sigarg[0];
      pstress_new[idx](1,1) = sigarg[1];
      pstress_new[idx](2,2) = sigarg[2];
      pstress_new[idx](0,1) = sigarg[3];
      pstress_new[idx](1,0) = sigarg[3];
      pstress_new[idx](2,1) = sigarg[4];
      pstress_new[idx](1,2) = sigarg[4];
      pstress_new[idx](2,0) = sigarg[5];
      pstress_new[idx](0,2) = sigarg[5];

#if 0
      cout << pstress_new[idx] << endl;
#endif

      c_dil = sqrt(USM/rho_cur);

      // Compute the strain energy for all the particles
      Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

      double e = (D(0,0)*AvgStress(0,0) +
                  D(1,1)*AvgStress(1,1) +
                  D(2,2)*AvgStress(2,2) +
              2.*(D(0,1)*AvgStress(0,1) +
                  D(0,2)*AvgStress(0,2) +
                  D(1,2)*AvgStress(1,2))) * pvolume_new[idx]*delT;

      se += e;

      // Compute wave speed at each particle, store the maximum
      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(UI[0]/rho_cur);
        Matrix3 D=(velGrad + velGrad.Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);

    delete interpolator;
  }
}

void Kayenta::carryForward(const PatchSubset* patches,
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
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void Kayenta::addComputesAndRequires(Task* task,
                                     const MPMMaterial* matl,
                                     const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);
}

void Kayenta::addComputesAndRequires(Task*,
                                     const MPMMaterial*,
                                     const PatchSet*,
                                     const bool ) const
{
}

double Kayenta::computeRhoMicroCM(double pressure,
                                  const double p_ref,
                                  const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double bulk = d_initialData.K;

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 1
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR Kayenta"
       << endl;
#endif
}

void Kayenta::computePressEOSCM(double rho_cur, double& pressure,
                                double p_ref,
                                double& dp_drho,      double& tmp,
                                const MPMMaterial* matl)
{

  //double G = d_initialData.G;
  double bulk = d_initialData.K;
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = bulk/rho_cur;  // speed of sound squared

#if 1
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Kayenta"
       << endl;
#endif
}

double Kayenta::getCompressibility()
{
  return 1.0/d_initialData.K;
}

void
Kayenta::getInputParameters(ProblemSpecP& ps)
{
  ps->require("B0",UI[0]);              // initial bulk modulus (stress)
  ps->getWithDefault("B1",UI[1],0.0);   // nonlinear bulk mod param (stress)
  ps->getWithDefault("B2",UI[2],0.0);   // nonlinear bulk mod param (stress)
  ps->getWithDefault("B3",UI[3],0.0);   // nonlinear bulk mod param (stress)
  ps->getWithDefault("B4",UI[4],0.0);   // nonlinear bulk mod param (dim. less)

  ps->require("G0",UI[5]);              // initial shear modulus (stress)
  ps->getWithDefault("G1",UI[6],0.0);   // nonlinear shear mod param (dim. less)
  ps->getWithDefault("G2",UI[7],0.0);   // nonlinear shear mod param (1/stress)
  ps->getWithDefault("G3",UI[8],0.0);   // nonlinear shear mod param (stress)
  ps->getWithDefault("G4",UI[9],0.0);   // nonlinear shear mod param (dim. less)

  ps->getWithDefault("RJS",UI[10],0.0); // joint spacing (iso. joint set) 
                                        // (length)
  ps->getWithDefault("RKS",UI[11],0.0); // joint shear stiffness (iso. case)                                            // (stress/length)
  ps->getWithDefault("RKN",UI[12],0.0); // joint normal stiffness (iso. case) 
                                        // (stress/length)

  ps->getWithDefault("A1",UI[13],0.0);  // meridional yld prof param (stress)
  ps->getWithDefault("A2",UI[14],0.0);  // meridional yld prof param (1/stress)
  ps->getWithDefault("A3",UI[15],0.0);  // meridional yld prof param (stress)
  ps->getWithDefault("A4",UI[16],0.0);  // meridional yld prof param (dim. less)

  ps->getWithDefault("P0",UI[17],0.0);  // init hydrostatic crush press (stress)
  ps->getWithDefault("P1",UI[18],0.0);  // crush curve parameter (1/stress)
  ps->getWithDefault("P2",UI[19],0.0);  // crush curve parameter (1/stress^2)
  ps->getWithDefault("P3",UI[20],0.0);  // crush curve parameter (strain)

  ps->getWithDefault("CR",UI[21],0.0);  // cap curvature parameter (dim. less)
  ps->getWithDefault("RK",UI[22],0.0);  // TXE/TXC strength ratio (dim. less)
  ps->getWithDefault("RN",UI[23],0.0);  // TXE/TXC strength ratio (stress)
  ps->getWithDefault("HC",UI[24],0.0);  // kinematic hardening modulus (stress)

  ps->getWithDefault("CTI1",UI[25],0.0);// Tension I1 cut-off (stress)
  ps->getWithDefault("CTPS",UI[26],0.0);// Tension prin. stress cut-off (stress)

  ps->getWithDefault("T1",UI[27],0.0);  // rate dep. primary relax. time (time)
  ps->getWithDefault("T2",UI[28],0.0);  // rate dep. nonlinear param (1/time)
  ps->getWithDefault("T3",UI[29],0.0);  // rate dep. nonlinear param (dim. less)
  ps->getWithDefault("T4",UI[30],0.0);  // not used (1/time)
  ps->getWithDefault("T5",UI[31],0.0);  // not used (stress)
  ps->getWithDefault("T6",UI[32],0.0);  // rate dep. nonlinear param (time)
  ps->getWithDefault("T7",UI[33],0.0);  // rate dep. nonlinear param (1/stress)

  ps->getWithDefault("J3TYPE",UI[34],0.0);// octahedral profile shape option
                                          // (dim. less)
  ps->getWithDefault("A2PF",UI[35],0.0);// flow potential analog of A2
  ps->getWithDefault("A4PF",UI[36],0.0);// flow potential analog of A4
  ps->getWithDefault("CRPF",UI[37],0.0);// flow potential analog of CR
  ps->getWithDefault("RKPF",UI[38],0.0);// flow potential analog of RK
  ps->getWithDefault("SUBX",UI[39],0.0);// subcycle control exponent (dim. less)

}

void
Kayenta::initializeLocalMPMLabels()
{
  ISVNames.resize(d_NINSV);
  
  ISVNames[1] ="KAPPA";
  ISVNames[2] ="INDEX";
  ISVNames[3] ="EQDOT";
  ISVNames[4] ="I1";
  ISVNames[5] ="ROOTJ2";
  ISVNames[6] ="ALXX";
  ISVNames[7] ="ALYY";
  ISVNames[8] ="ALZZ";
  ISVNames[9] ="ALXY";
  ISVNames[10]="ALYZ";
  ISVNames[11]="ALXZ";
  ISVNames[12]="GFUN";
  ISVNames[13]="EQPS";
  ISVNames[14]="EQPV";
  ISVNames[15]="EL0";
  ISVNames[16]="HK";
  ISVNames[17]="EVOL";
  ISVNames[18]="BACKRN";
  ISVNames[19]="CRACK";
  ISVNames[20]="SHEAR";
  ISVNames[21]="YIELD";
  ISVNames[22]="LODE";
  ISVNames[23]="QSSIGXX";
  ISVNames[24]="QSSIGYY";
  ISVNames[25]="QSSIGZZ";
  ISVNames[26]="QSSIGXY";
  ISVNames[27]="QSSIGYZ";
  ISVNames[28]="QSSIGXZ";
  ISVNames[29]="DSCP";
  ISVNames[30]="QSEL";
  ISVNames[31]="QSBSXX";
  ISVNames[32]="QSBSYY";
  ISVNames[33]="QSBSZZ";
  ISVNames[34]="QSBSXY";
  ISVNames[35]="QSBSYZ";
  ISVNames[36]="QSBSXZ";

  for(int i=1;i<=d_NINSV;i++){
    ISVLabels.push_back(VarLabel::create(ISVNames[i],
                          ParticleVariable<double>::getTypeDescription()));
    ISVLabels_preReloc.push_back(VarLabel::create(ISVNames[i]+"+",
                          ParticleVariable<double>::getTypeDescription()));
  }
}
