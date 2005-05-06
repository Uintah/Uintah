#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElastic.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h> 
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

HypoElastic::HypoElastic(ProblemSpecP& ps, MPMLabel* Mlb, MPMFlags* Mflag)
  : ConstitutiveModel(Mlb,Mflag)
{
  ps->require("G",d_initialData.G);
  ps->require("K",d_initialData.K);

  // Thermal expansion coefficient 
  d_initialData.alpha=0.0;
  ps->get("alpha",d_initialData.alpha); // for thermal stress 

  if (flag->d_fracture) {
    // Read in fracture criterion and the toughness curve
    ProblemSpecP curve_ps = ps->findBlock("fracture_toughness_curve");
    if(curve_ps!=0) {
      crackPropagationCriterion="max_hoop_stress";        
      curve_ps->get("crack_propagation_criterion",crackPropagationCriterion);     

      if(crackPropagationCriterion!="max_hoop_stress" && 
         crackPropagationCriterion!="max_principal_stress" &&
         crackPropagationCriterion!="max_energy_release_rate" &&
         crackPropagationCriterion!="strain_energy_density" &&
         crackPropagationCriterion!="empirical_criterion") {
        cout << "!!! Undefinded crack propagation criterion: "
             << crackPropagationCriterion 
             << " for hypo-elastic material. Program terminated."
             << endl;
        exit(1);       
      }     

      if(crackPropagationCriterion=="empirical_criterion") {
        // Get parameters p & q in the fracture locus equation
        // (KI/Ic)^p+(KII/KIIc)^q=1 and KIIc=r*KIc       
        p=q=2.0;  // Default elliptical fracture locus  
        r=-1.;
        curve_ps->get("p",p);
        curve_ps->get("q",q);
        curve_ps->get("r",r);
      }
            
      for(ProblemSpecP child_ps=curve_ps->findBlock("point"); child_ps!=0; 
          child_ps=child_ps->findNextBlock("point")) {
        double Vc,KIc,KIIc;
        child_ps->get("Vc",Vc);
        child_ps->get("KIc",KIc);
        if(r<0.) { // Input KIIc manually
          child_ps->get("KIIc",KIIc);
        }
        else { // The ratio of KIIc to KIc is a constant (r) 
          KIIc=r*KIc;
        }       
        d_initialData.Kc.push_back(Vector(Vc,KIc,KIIc));
      }           
    }
  }
}

HypoElastic::HypoElastic(const HypoElastic* cm)
{
  lb = cm->lb;
  flag = cm->flag;
  NGN = cm->NGN;
  d_initialData.G = cm->d_initialData.G;
  d_initialData.K = cm->d_initialData.K;
  d_initialData.alpha = cm->d_initialData.alpha; // for thermal stress
  if (flag->d_fracture)
    d_initialData.Kc = cm->d_initialData.Kc;
}

HypoElastic::~HypoElastic()
{
  // Destructor

}

void HypoElastic::initializeCMData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  if (flag->d_fracture) {
    // Put stuff in here to initialize each particle's
    // constitutive model parameters and deformationMeasure
    Matrix3 Identity, zero(0.);
    Identity.Identity();

    ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

    // for J-Integral
    ParticleVariable<Matrix3> pdispGrads;
    ParticleVariable<double>  pstrainEnergyDensity;
    new_dw->allocateAndPut(pdispGrads, lb->pDispGradsLabel, pset);
    new_dw->allocateAndPut(pstrainEnergyDensity, lb->pStrainEnergyDensityLabel, 
                           pset);

    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){
      pdispGrads[*iter] = zero;
      pstrainEnergyDensity[*iter] = 0.0;
    }
  }

  computeStableTimestep(patch, matl, new_dw);
}


void HypoElastic::allocateCMDataAddRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches,
                                            MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);

  // Add requires local to this model
  Ghost::GhostType  gnone = Ghost::None;
  if (flag->d_fracture) {
    task->requires(Task::NewDW, lb->pDispGradsLabel_preReloc, matlset, gnone);
    task->requires(Task::NewDW, lb->pStrainEnergyDensityLabel_preReloc,
                   matlset, gnone);
  }
}


void HypoElastic::allocateCMDataAdd(DataWarehouse* new_dw,
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
  if (flag->d_fracture) {
    ParticleVariable<Matrix3> pdispGrads;
    constParticleVariable<Matrix3> o_dispGrads;
    ParticleVariable<double>  pstrainEnergyDensity;
    constParticleVariable<double>  o_strainEnergyDensity;
    // for J-Integral
    new_dw->allocateTemporary(pdispGrads, addset);
    new_dw->get(o_dispGrads,lb->pDispGradsLabel_preReloc,delset);
    new_dw->allocateTemporary(pstrainEnergyDensity,addset);
    new_dw->get(o_strainEnergyDensity,lb->pStrainEnergyDensityLabel_preReloc,
                delset);

    ParticleSubset::iterator o, n = addset->begin();
    for (o=delset->begin(); o != delset->end(); o++, n++) {
      pdispGrads[*n] = o_dispGrads[*o];
      pstrainEnergyDensity[*n] = o_strainEnergyDensity[*o];
    }

    (*newState)[lb->pDispGradsLabel]=pdispGrads.clone();
    (*newState)[lb->pStrainEnergyDensityLabel]=pstrainEnergyDensity.clone();
  }
}

void HypoElastic::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  if (flag->d_fracture) {
    from.push_back(lb->pDispGradsLabel);
    from.push_back(lb->pStrainEnergyDensityLabel);
    to.push_back(lb->pDispGradsLabel_preReloc);
    to.push_back(lb->pStrainEnergyDensityLabel_preReloc);
  }
}

void HypoElastic::computeStableTimestep(const Patch* patch,
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

  double G = d_initialData.G;
  double bulk = d_initialData.K;
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
    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(delT_new)), 
                lb->delTLabel);
}

void HypoElastic::computeStressTensor(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  double rho_orig = matl->getInitialDensity();
  for(int p=0;p<patches->size();p++){
    double se = 0.0;
    const Patch* patch = patches->get(p);
    //
    //  FIX  To do:  Read in table for vres
    //               Obtain and modify particle temperature (deg K)
    //

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());

    Matrix3 velGrad,deformationGradientInc,Identity,zero(0.),One(1.);
    double c_dil=0.0,Jinc;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    double onethird = (1.0/3.0);

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    //double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;

    int dwi = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3> pstress_new;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<double> pmass, pvolume, ptemperature;
    ParticleVariable<double> pvolume_deformed;
    constParticleVariable<Vector> pvelocity, psize;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;
    // for thermal stress
    constParticleVariable<double> pTempPrevious, pTempCurrent; 

    Ghost::GhostType  gac   = Ghost::AroundCells;

    old_dw->get(psize,             lb->pSizeLabel,               pset);
    
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    // for thermal stress
    old_dw->get(pTempPrevious,       lb->pTempPreviousLabel,       pset); 
    new_dw->get(pTempCurrent,        lb->pTempCurrentLabel,        pset); 

    new_dw->get(gvelocity,lb->gVelocityLabel, dwi,patch, gac, NGN);

    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    constNCVariable<Vector> Gvelocity;
    constParticleVariable<Short27> pgCode;
    constParticleVariable<Matrix3> pdispGrads;
    constParticleVariable<double>  pstrainEnergyDensity;
    ParticleVariable<Matrix3> pvelGrads;
    ParticleVariable<Matrix3> pdispGrads_new;
    ParticleVariable<double> pstrainEnergyDensity_new;
    if (flag->d_fracture) {
      new_dw->get(Gvelocity,lb->GVelocityLabel, dwi, patch, gac, NGN);
      new_dw->get(pgCode,              lb->pgCodeLabel,              pset);
      old_dw->get(pdispGrads,          lb->pDispGradsLabel,          pset);
      old_dw->get(pstrainEnergyDensity,lb->pStrainEnergyDensityLabel,pset);
      new_dw->allocateAndPut(pvelGrads,  lb->pVelGradsLabel,  pset);
          
      new_dw->allocateAndPut(pdispGrads_new, lb->pDispGradsLabel_preReloc, pset);
      new_dw->allocateAndPut(pstrainEnergyDensity_new,
                             lb->pStrainEnergyDensityLabel_preReloc, pset);
    }

    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,   pset);
    new_dw->allocateAndPut(deformationGradient_new,
                           lb->pDeformationMeasureLabel_preReloc, pset);
 
    // Allocate variable to store internal heating rate
    ParticleVariable<double> pIntHeatRate;
    new_dw->allocateAndPut(pIntHeatRate, lb->pInternalHeatRateLabel_preReloc, 
                           pset);

    double G    = d_initialData.G;
    double bulk = d_initialData.K;
    double alpha = d_initialData.alpha;   // for thermal stress    

    for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pIntHeatRate[idx] = 0.0;

      // Get the node indices that surround the cell
      interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx]);

       Vector gvel;
       velGrad.set(0.0);
       for(int k = 0; k < flag->d_8or27; k++) {
         if (flag->d_fracture) {
           if(pgCode[idx][k]==1) gvel = gvelocity[ni[k]];
           if(pgCode[idx][k]==2) gvel = Gvelocity[ni[k]];
         } else
           gvel = gvelocity[ni[k]];
         for (int j = 0; j<3; j++){
           for (int i = 0; i<3; i++) {
             velGrad(i,j)+=gvel[i] * d_S[k][j] * oodx[j];
             if (flag->d_fracture)
               pvelGrads[idx](i,j)  = velGrad(i,j);
            }
          }
      }

      // Rate of particle temperature change for thermal stress
      double ptempRate=(pTempCurrent[idx]-pTempPrevious[idx])/delT; 
      // Calculate rate of deformation D, and deviatoric rate DPrime,
      // including effect of thermal strain
      Matrix3 D = (velGrad + velGrad.Transpose())*.5-Identity*alpha*ptempRate;
      Matrix3 DPrime = D - Identity*onethird*D.Trace();

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
      pvolume_deformed[idx]=Jinc*pvolume[idx];

      // Compute the local sound speed
      double rho_cur = rho_orig/J;
      c_dil = sqrt((bulk + 4.*G/3.)/rho_cur);
      // 
      // This is the (updated) Cauchy stress
      pstress_new[idx] = pstress[idx] + 
                         (DPrime*2.*G + Identity*bulk*D.Trace())*delT;

      // Add bulk viscosity
      /*
      if (flag->d_artificial_viscosity) {
        double Dkk = D.Trace();
        double c_bulk = sqrt(bulk/rho_cur);
        double q = artificialBulkViscosity(Dkk, c_bulk, rho_cur, dx_ave);
        pstress_new[idx] -= Identity*q;
      }
      */

      // Compute the strain energy for all the particles
      Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

      double e = (D(0,0)*AvgStress(0,0) +
                  D(1,1)*AvgStress(1,1) +
                  D(2,2)*AvgStress(2,2) +
               2.*(D(0,1)*AvgStress(0,1) +
                   D(0,2)*AvgStress(0,2) +
                   D(1,2)*AvgStress(1,2))) * pvolume_deformed[idx]*delT;

      se += e;

      if (flag->d_fracture) {
        // Update particle displacement gradients
        pdispGrads_new[idx] = pdispGrads[idx] + velGrad * delT;
        // Update particle strain energy density 
        pstrainEnergyDensity_new[idx] = pstrainEnergyDensity[idx] + 
                                         e/pvolume_deformed[idx];
      }

      // Compute wave speed at each particle, store the maximum
      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(delT_new)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);

    delete interpolator;
  }
}

void HypoElastic::carryForward(const PatchSubset* patches,
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
    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(1.e10)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

// Convert J-integral into stress intensity factors for hypoelastic materials
void 
HypoElastic::ConvertJToK(const MPMMaterial* matl,const Vector& J,
                     const double& C,const Vector& V,Vector& SIF)
{                    
  /* J--J integral, C--Crack velocity, V--COD near crack tip
     in local coordinates. */ 
     
  double J1,CC,V1,V2;
  
  J1=J.x();                           // total energy release rate
  V1=V.y();  V2=V.x();                // V1--opening COD, V2--sliding COD
  CC=C*C;                             // square of crack propagating velocity
  
  // get material properties
  double rho_orig,G,K,v,k;
  rho_orig=matl->getInitialDensity();
  G=d_initialData.G;                  // shear modulus
  K=d_initialData.K;                  // bulk modulus
  v=0.5*(3.*K-2.*G)/(3*K+G);          // Poisson ratio
  string stressState="planeStress";   // Plane stress
  k = (stressState=="planeStress")? (3.-v)/(1.+v) : (3.-4.*v); 

  double Cs2,Cd2,D,B1,B2,A1,A2;
  if(sqrt(CC)<1.e-16) {               // for static crack
    B1=B2=1.;
    A1=A2=(k+1.)/4.;
  }
  else {                              // for dynamic crack
    Cs2=G/rho_orig;
    Cd2=(k+1.)/(k-1.)*Cs2;

    if(CC>Cs2) CC=Cs2;
    
    B1=sqrt(1.-CC/Cd2);
    B2=sqrt(1.-CC/Cs2);
    D=4.*B1*B2-(1.+B2*B2)*(1.+B2*B2);
    A1=B1*(1.-B2*B2)/D;
    A2=B2*(1.-B2*B2)/D;
  }

  double COD2,KI,KII;
  COD2=V1*V1*B2+V2*V2*B1;
  if(sqrt(COD2)<1.e-32) {            // COD=0
    KI  = 0.;
    KII = 0.;
  }
  else {
    KI =V1*sqrt(2.*G*B2*fabs(J1)/A1/COD2);
    KII=V2*sqrt(2.*G*B1*fabs(J1)/A2/COD2);
  }
  SIF=Vector(KI,KII,0.);
}

// Detect if crack propagates and the propagation direction
// for FRACTURE
short
HypoElastic::CrackPropagates(const double& Vc,const double& KI,
                             const double& KII, double& theta)
{
  /* Task 1: Determine fracture toughness KIc at velocity Vc
  */
  // Dynamic fracture toughness Kc(Vc,KIc,KIIC) 
  vector<Vector> Kc = d_initialData.Kc;
  int num = (int) Kc.size();
  double KIc=-1., KIIc=-1.0;
  if(Vc<=Kc[0].x()) { // Beyond the left bound
    KIc=Kc[0].y();
    KIIc=Kc[0].z();
  }
  else if(Vc>=Kc[num-1].x()) { // Beyond the right bound
    KIc=Kc[num-1].y();
    KIIc=Kc[num-1].z();
  }
  else { // In between 
    for(int i=0; i<num-1;i++) {
      double Vi=Kc[i].x();
      double Vj=Kc[i+1].x();
      if(Vc>=Vi && Vc<Vj) {
        double KIi=Kc[i].y();
        double KIj=Kc[i+1].y();
        double KIIi=Kc[i].z();
        double KIIj=Kc[i+1].z();
        KIc=KIi+(KIj-KIi)*(Vc-Vi)/(Vj-Vi);
        KIIc=KIIi+(KIIj-KIIi)*(Vc-Vi)/(Vj-Vi);  
        break;
      }
    } // End of loop over i
  }

  /* Task 2: Determine crack propagation direction (theta) and 
             the equivalent stress intensity factor (Kq)
  */         
  double Kq=-9e32;
  if(crackPropagationCriterion=="max_hoop_stress" ||
     crackPropagationCriterion=="max_energy_release_rate") {
    // Crack propagation direction        
    double sinTheta,cosTheta,value;
    if(KI==0.0 || (KI!=0. && fabs(KII/KI)>1000.)) { // Pure mode II
      cosTheta = 1./3.;
      sinTheta = (KII>=0.) ? -sqrt(8./9.) : sqrt(8./9.);
    }
    else {  // Mixed mode or pure mode I 
      double R=KII/KI;      
      cosTheta=(3.*R*R+sqrt(1.+8.*R*R))/(1.+9.*R*R);
      value=fabs(R*(3.*cosTheta-1.));
      sinTheta = (KII>=0.)? -value : value;
    }
    theta=asin(sinTheta);
      
    // Equivalent stress intensity
    double ct=cos(theta/2.);
    double st=sin(theta/2.);
    Kq=KI*pow(ct,3)-3*KII*ct*ct*st;
  } // End of max_hoop_stress criterion
 
  if(crackPropagationCriterion=="max_principal_stress") {
    if(KII==0. || (KII!=0. && fabs(KI/KII)>1000.)) { // Pure mode I
      theta=0.;
      Kq=KI; 
    }       
    else { // Mixed mode or pure mode II
      double R=KI/KII;      
      int sign = (KII>0.)? -1 : 1;
      double tanTheta2=(R+sign*sqrt(R*R+8.))/4.;
      double theta2=atan(tanTheta2);
      double ct=cos(theta2);
      double st=sin(theta2);
      Kq=KI*pow(ct,3)-3*KII*ct*ct*st;
      theta=2*theta2;
    }                             
  } // End of max_principal_stress criterion
  
  if(crackPropagationCriterion=="strain_energy_density") {
    // Calculate parameter k
    double k;
    string stressState="planeStress";     // Plane stress
    double G=d_initialData.G;             // Shear modulus
    double K=d_initialData.K;             // Bulk modulus
    double v=0.5*(3*K-2*G)/(3*K+G);     // Poisson ratio
    k = (stressState=="planeStress")? (3.-v)/(1.+v) : (3.-4.*v);
    
    // Crack propagation direction
    if(KII==0.0 || (KII!=0. && fabs(KI/KII)>1000.)) { // Pure mode I
      theta=0.0;
      Kq=KI;
    }
    else { // Mixed mode or pure mode II
      theta=CrackPropagationAngleFromStrainEnergyDensityCriterion(k,KI,KII);
      // Equivalent stress intensity
      double ct=cos(theta),st=sin(theta);
      double a11=(1+ct)*(k-ct);
      double a12=st*(2*ct-k+1);
      double a22=(k+1)*(1-ct)+(1+ct)*(3*ct-1);
      Kq=sqrt((a11*KI*KI+2*a12*KI*KII+a22*KII*KII)/2/(k-1)); 
    }  
  }  // End of strain_energy_density criterion       

  if(crackPropagationCriterion=="empirical_criterion") {
    if(KII==0. || (KII!=0. && fabs(KI/KII)>1000.)) { // Pure mode I
      theta=0.;
      Kq=KI;
    }
    else { // For mixed mode or pure mode II, use maximum pricipal criterion to
           // determine the crack propagation direction and the emprical
           // criterion to determine if crack will propagate.
      double R=KI/KII;
      int sign = (KII>0.)? -1 : 1;
      double tanTheta2=(R+sign*sqrt(R*R+8.))/4.;
      theta=2*atan(tanTheta2);
      Kq=(pow(KI/KIc,p)+pow(KII/KIIc,q))*KIc;
    }
  } // End of empirical_criterion

  if(Kq>=KIc)
    return 1;
  else
    return 0;
}

// Obtain crack propagation angle numerically from strain energy density criterion
// for FRACTURE
double
HypoElastic::CrackPropagationAngleFromStrainEnergyDensityCriterion(const double& k,
                const double& KI, const double& KII)
{
  double errF=1.e-6,errV=1.e-2,PI=3.141592654;
  double a,b,c,fa,fb,fc;
  
  double A=-PI, B=PI;   // The region of the roots 
  int n=36;             // Divide [A,B] into n intervals
  double h=(B-A)/n;     // Subinterval length
  
  double theta=0.0;
  double theta0=atan(KI/KII);
  vector<double> root;  // Store the solutions of the equation
  // Solve the equation numerically
  for(int i=0; i<n; i++) { // Loop over the whole interval [A,B]
    a=A+i*h;
    b=A+(i+1)*h;
    fa=(k-1)*sin(a-2*theta0)-2*sin(2*(a-theta0))-sin(2*a);
    fb=(k-1)*sin(b-2*theta0)-2*sin(2*(b-theta0))-sin(2*b);
    
    // Find the root in [a,b)
    if(fabs(fa)<errF) { // Where f(a)=0
      root.push_back(a);
    }
    else if(fa*fb<0.) { // There is a root in (a,b)
      double cp=2*B;    // Set the value beyond [A,B]      
      for(int j=0; j<32768; j++) { // 32768=2^15 (a big int) 
        c=b-(a-b)*fb/(fa-fb);
        fc=(k-1)*sin(c-2*theta0)-2*sin(2*(c-theta0))-sin(2*c);
        if(fabs(fc)<errF || fabs(c-cp)<errV) { // c is the root
          root.push_back(c);
          break;
        }
        else { // Record the cross point with axis x
          cp=c;
        }  
          
        // Narrow the region of the root
        if(fc*fa<0.) { // The root is in (a,c)
          fb=fc; b=c;
        }
        else if(fc*fb<0.) { // The root is in (c,b)
          fa=fc; a=c;
        }
      } // End of loop over j
    } // End of if(fa*fb<0.)
  } // End of loop over i
    
  // Select the direction from the solutions 
  // along which there exists the minimum strain energy density   
  int count=0;
  double S0=0.0;
  for(int i=0;i<(int)root.size();i++) {
    double r=root[i];       

    // The signs of propagation angle and KII must be opposite 
    if(KII*r>0.) continue;

    // Calculate the second derivative of the strain energy density
    double sr=sin(r),cr=cos(r),sr2=sin(2*r),cr2=cos(2*r);
    double dsdr2=KI*KI*((1-k)*cr+2*cr2)-2*KI*KII*(4*sr2+(1-k)*sr)+
                 KII*KII*((k-1)*cr-6*cr2); 
    if(dsdr2>0.) { 
      // Determine propagation angle by comparison of strain energy density. 
      // Along the angle there exists the minimum strain energy density. 
      double S=(1+cr)*(k-cr)*KI*KI+2*sr*(2*cr-k+1)*KI*KII+
               ((k+1)*(1-cr)+(1+cr)*(3*cr-1))*KII*KII; 
      if(count==0 || (count>0 && S<S0)) {
        theta=r;  
        S0=S;
        count++;
      } 
    }
  } // Enf of loop over i
  root.clear();

  return theta;
}

void HypoElastic::addComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);
  
  Ghost::GhostType gnone = Ghost::None;
  // for thermal stress
  task->requires(Task::OldDW, lb->pTempPreviousLabel, matlset, gnone); 
  task->requires(Task::NewDW, lb->pTempCurrentLabel,  matlset, gnone); 

  // Other constitutive model and input dependent computes and requires
  if (flag->d_fracture) {
    Ghost::GhostType  gnone = Ghost::None;
    task->requires(Task::OldDW, lb->pDispGradsLabel,           matlset, gnone);
    task->requires(Task::OldDW, lb->pStrainEnergyDensityLabel, matlset, gnone);
    
    task->computes(lb->pDispGradsLabel_preReloc,             matlset);
    task->computes(lb->pVelGradsLabel,                       matlset);
    task->computes(lb->pStrainEnergyDensityLabel_preReloc,   matlset);
  }
}

void 
HypoElastic::addComputesAndRequires(Task*,const MPMMaterial*, const PatchSet*,
                                    const bool ) const
{
}

double HypoElastic::computeRhoMicroCM(double pressure,
                                      const double p_ref,
                                      const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  //double p_ref=101325.0;
  double p_gauge = pressure - p_ref;
  double rho_cur;
  //double G = d_initialData.G;
  double bulk = d_initialData.K;

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 0
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR HypoElastic"
       << endl;
#endif
}

void HypoElastic::computePressEOSCM(double rho_cur, double& pressure,
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

#if 0
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR HypoElastic"
       << endl;
#endif
}

double HypoElastic::getCompressibility()
{
  return 1.0/d_initialData.K;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {

#if 0
static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(HypoElastic::StateData), sizeof(double)*2);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 2, 2, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(HypoElastic::StateData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew
        TypeDescription(TypeDescription::Other,
                        "HypoElastic::StateData", true, &makeMPI_CMData);
   }
   return td;
}
#endif

} // End namespace Uintah
