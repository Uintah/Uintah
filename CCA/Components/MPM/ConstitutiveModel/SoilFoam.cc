#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/SoilFoam.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Math/Short27.h> // for Fracture
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>

#include <sci_values.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;

using namespace Uintah;
using namespace SCIRun;

// Material Constants are C1, C2 and PR (poisson's ratio).  
// The shear modulus = 2(C1 + C2).

SoilFoam::SoilFoam(ProblemSpecP& ps, MPMFlags* Mflag) 
  : ConstitutiveModel(Mflag)
{

  ps->require("elastic_shear",d_initialData.G);
  ps->require("bulk",d_initialData.bulk);
  ps->require("a0",d_initialData.a0);
  ps->require("a1",d_initialData.a1);
  ps->require("a2",d_initialData.a2);
  ps->require("pc",d_initialData.pc);
  ps->require("eps0",d_initialData.eps[0]);
  ps->require("eps1",d_initialData.eps[1]);
  ps->require("eps2",d_initialData.eps[2]);
  ps->require("eps3",d_initialData.eps[3]);
  ps->require("eps4",d_initialData.eps[4]);
  ps->require("eps5",d_initialData.eps[5]);
  ps->require("eps6",d_initialData.eps[6]);
  ps->require("eps7",d_initialData.eps[7]);
  ps->require("eps8",d_initialData.eps[8]);
  ps->require("eps9",d_initialData.eps[9]);
  ps->require("p0",d_initialData.p[0]);
  ps->require("p1",d_initialData.p[1]);
  ps->require("p2",d_initialData.p[2]);
  ps->require("p3",d_initialData.p[3]);
  ps->require("p4",d_initialData.p[4]);
  ps->require("p5",d_initialData.p[5]);
  ps->require("p6",d_initialData.p[6]);
  ps->require("p7",d_initialData.p[7]);
  ps->require("p8",d_initialData.p[8]);
  ps->require("p9",d_initialData.p[9]);
  int i;
  for(i=0; i<9; i++){
    slope[i] = (d_initialData.p[i+1] - d_initialData.p[i])/(d_initialData.eps[i+1] - d_initialData.eps[i]);
  }

  sv_minLabel = VarLabel::create("p.sv_minLabel",
                ParticleVariable<double>::getTypeDescription());
  sv_minLabel_preReloc = VarLabel::create("p.sv_minLabel+",
                ParticleVariable<double>::getTypeDescription());
  p_sv_minLabel = VarLabel::create("p.p_sv_minLabel",
                ParticleVariable<double>::getTypeDescription());
  p_sv_minLabel_preReloc = VarLabel::create("p.p_sv_minLabel+",
                ParticleVariable<double>::getTypeDescription());
  csv_minLabel         = VarLabel::create( "c.sv_min",
                     CCVariable<double>::getTypeDescription() );
}

SoilFoam::SoilFoam(const SoilFoam* cm)
  : ConstitutiveModel(cm)
{
  d_initialData.G = cm->d_initialData.G;
  d_initialData.bulk = cm->d_initialData.bulk;
  d_initialData.a0 = cm->d_initialData.a0;
  d_initialData.a1 = cm->d_initialData.a1;
  d_initialData.a2 = cm->d_initialData.a2;
  d_initialData.pc = cm->d_initialData.pc;
  int i;
  for(i=0; i<10; i++){
     d_initialData.eps[i] = cm->d_initialData.eps[i];
     d_initialData.p[i] = cm->d_initialData.p[i];
  }
  for(i=0; i<9; i++) slope[i] = cm->slope[i];

  sv_minLabel = VarLabel::create("p.sv_minLabel",
                ParticleVariable<double>::getTypeDescription());
  sv_minLabel_preReloc = VarLabel::create("p.sv_minLabel+",
                ParticleVariable<double>::getTypeDescription());
  p_sv_minLabel = VarLabel::create("p.p_sv_minLabel",
                ParticleVariable<double>::getTypeDescription());
  p_sv_minLabel_preReloc = VarLabel::create("p.p_sv_minLabel+",
                ParticleVariable<double>::getTypeDescription());
  csv_minLabel         = VarLabel::create( "c.sv_min",
                     CCVariable<double>::getTypeDescription() );
}

void SoilFoam::addParticleState(std::vector<const VarLabel*>& from,
                                        std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(sv_minLabel);
  from.push_back(p_sv_minLabel);

  to.push_back(sv_minLabel_preReloc);
  to.push_back(p_sv_minLabel_preReloc);
}

SoilFoam::~SoilFoam()
{
  VarLabel::destroy(sv_minLabel);
  VarLabel::destroy(sv_minLabel_preReloc);
  VarLabel::destroy(p_sv_minLabel);
  VarLabel::destroy(p_sv_minLabel_preReloc);
  VarLabel::destroy(csv_minLabel);
}

void SoilFoam::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","soil_foam");
  }
    
  cm_ps->appendElement("elastic_shear",d_initialData.G);
  cm_ps->appendElement("bulk",d_initialData.bulk);
  cm_ps->appendElement("a0",d_initialData.a0);
  cm_ps->appendElement("a1",d_initialData.a1);
  cm_ps->appendElement("a2",d_initialData.a2);
  cm_ps->appendElement("pc",d_initialData.pc);
  cm_ps->appendElement("eps0",d_initialData.eps[0]);
  cm_ps->appendElement("eps1",d_initialData.eps[1]);
  cm_ps->appendElement("eps2",d_initialData.eps[2]);
  cm_ps->appendElement("eps3",d_initialData.eps[3]);
  cm_ps->appendElement("eps4",d_initialData.eps[4]);
  cm_ps->appendElement("eps5",d_initialData.eps[5]);
  cm_ps->appendElement("eps6",d_initialData.eps[6]);
  cm_ps->appendElement("eps7",d_initialData.eps[7]);
  cm_ps->appendElement("eps8",d_initialData.eps[8]);
  cm_ps->appendElement("eps9",d_initialData.eps[9]);
  cm_ps->appendElement("p0",d_initialData.p[0]);
  cm_ps->appendElement("p1",d_initialData.p[1]);
  cm_ps->appendElement("p2",d_initialData.p[2]);
  cm_ps->appendElement("p3",d_initialData.p[3]);
  cm_ps->appendElement("p4",d_initialData.p[4]);
  cm_ps->appendElement("p5",d_initialData.p[5]);
  cm_ps->appendElement("p6",d_initialData.p[6]);
  cm_ps->appendElement("p7",d_initialData.p[7]);
  cm_ps->appendElement("p8",d_initialData.p[8]);
  cm_ps->appendElement("p9",d_initialData.p[9]);
}

SoilFoam* SoilFoam::clone()
{
  return scinew SoilFoam(*this);
}

void 
SoilFoam::initializeCMData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double> sv_min, p_sv_min;
  new_dw->allocateAndPut(sv_min,    sv_minLabel,        pset);
  new_dw->allocateAndPut(p_sv_min,  p_sv_minLabel,      pset);

  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++){
    sv_min[*iter] = 0.0;
    p_sv_min[*iter] = 0.0;
  }

  computeStableTimestep(patch, matl, new_dw);

}

///////////////////////////////////////////////////////////////////////////
/*! Allocate data required during the conversion of failed particles 
    from one material to another */
///////////////////////////////////////////////////////////////////////////
void 
SoilFoam::allocateCMDataAddRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches ,
                                            MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);

  // Allocate other variables used in the conversion process

}


void SoilFoam::allocateCMDataAdd(DataWarehouse* new_dw,
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
}


void SoilFoam::computeStableTimestep(const Patch* patch,
                                             const MPMMaterial* matl,
                                             DataWarehouse* new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;
  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  double G = d_initialData.G;
  double bulk = d_initialData.bulk;

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     // Compute wave speed + particle velocity at each particle, 
     // store the maximum
     //double E = 9.0*bulk/(3.0*bulk/G + 1.0);
     c_dil = sqrt((bulk + 4.*G/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    if(delT_new < 1.e-12)
      // don't use adjustDelt here because of DBL_MAX
      new_dw->put(delt_vartype(DBL_MAX), lb->delTLabel);
    else
      new_dw->put(delt_vartype(d_sharedState->adjustDelt(patch->getLevel(), delT_new)), 
                  lb->delTLabel);
}

void SoilFoam::computeStressTensor(const PatchSubset* patches,
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
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    Matrix3 velGrad,deformationGradientInc,Identity,zero(0.),One(1.);
    double c_dil=0.0,Jinc;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    double onethird = (1.0/3.0);

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
    constParticleVariable<double> pmass, pvolume, ptemperature, sv_min, p_sv_min;
    ParticleVariable<double> pvolume_deformed, sv_min_new, p_sv_min_new;
    constParticleVariable<Vector> pvelocity, psize;
    constNCVariable<Vector> gvelocity;
    constCCVariable<double> csv_min;
    delt_vartype delT;
    // for thermal stress
    constParticleVariable<double> pTempPrevious, pTempCurrent; 

    Ghost::GhostType  gac   = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;

    old_dw->get(psize,             lb->pSizeLabel,               pset);
    
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(sv_min,              sv_minLabel,                  pset);
    old_dw->get(p_sv_min,            p_sv_minLabel,                pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

    new_dw->get(gvelocity,lb->gVelocityLabel, dwi,patch, gac, NGN);
    if(flag->d_with_ice)
      new_dw->get(csv_min,  csv_minLabel,       dwi,patch,gn, 0);

    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    constNCVariable<Vector> Gvelocity;
    constParticleVariable<Short27> pgCode;
    constParticleVariable<Matrix3> pdispGrads;
    constParticleVariable<double>  pstrainEnergyDensity;
    ParticleVariable<Matrix3> pvelGrads;
    ParticleVariable<Matrix3> pdispGrads_new;
    ParticleVariable<double> pstrainEnergyDensity_new;

    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,   pset);
    new_dw->allocateAndPut(sv_min_new,   sv_minLabel_preReloc,           pset);
    new_dw->allocateAndPut(p_sv_min_new, p_sv_minLabel_preReloc,         pset);
    new_dw->allocateAndPut(deformationGradient_new,
                           lb->pDeformationMeasureLabel_preReloc, pset);
 
    // Allocate variable to store internal heating rate
    ParticleVariable<double> pdTdt;
    new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel_preReloc, 
                           pset);

    double G    = d_initialData.G;
    double bulk = d_initialData.bulk;

    for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // Get the node indices that surround the cell
      interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx]);

       Vector gvel;
       velGrad.set(0.0);
       for(int k = 0; k < flag->d_8or27; k++) {
           gvel = gvelocity[ni[k]];
         for (int j = 0; j<3; j++){
           for (int i = 0; i<3; i++) {
             velGrad(i,j)+=gvel[i] * d_S[k][j] * oodx[j];
            }
          }
      }

      // Calculate rate of deformation D, and deviatoric rate DPrime,
      // including effect of thermal strain
      Matrix3 D = (velGrad + velGrad.Transpose())*.5;
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
      double vol_strain = log(J);//log(pvolume_deformed[idx]/(pmass[idx]/rho_orig));
      double pres_guage;
      double rho_cur = rho_orig/J;

      double maxvolstrain;
      if(flag->d_with_ice){
	IntVector c1(-1,-1,-1), c2(0,0,0);
        Point pt1 = getLevel(patches)->getCellPosition(c1), pt2 = getLevel(patches)->getCellPosition(c2);
	Vector x0 = 0.5*(pt1.asVector() + pt2.asVector());
	int i,j,k;
	i = int((px[idx].x() - x0.x())/dx.x());
	j = int((px[idx].y() - x0.y())/dx.y());
	k = int((px[idx].z() - x0.z())/dx.z());
        IntVector c3(i,j,k);
        maxvolstrain = csv_min[c3];
      }
      else{
	maxvolstrain = sv_min[idx];
      }

      // Traditional method for mat5
      if(vol_strain>maxvolstrain){//sv_min[idx]){
	pres_guage = p_sv_min[idx] - bulk*(vol_strain - maxvolstrain);//sv_min[idx]);
         if(pres_guage<d_initialData.pc) pres_guage = d_initialData.pc;
         p_sv_min_new[idx] = p_sv_min[idx];
         sv_min_new[idx] = maxvolstrain;//sv_min[idx];

         // Compute the local sound speed
         c_dil = sqrt((bulk + 4.*G/3.)/rho_cur);
      }else{
         int i1 = 0, i;
         for(i=1; i<9; i++){
            if(d_initialData.eps[i+1]<d_initialData.eps[i])
            if(vol_strain<d_initialData.eps[i]) i1 = i;
         }
         pres_guage = d_initialData.p[i1] + slope[i1]*(vol_strain - d_initialData.eps[i1]);
         if(pres_guage<d_initialData.pc) pres_guage = d_initialData.pc;
         p_sv_min_new[idx] = pres_guage;
         sv_min_new[idx] = vol_strain;
         // Compute the local sound speed
         c_dil = sqrt((-slope[i1] + 4.*G/3.)/rho_cur);
      }
      // 
      // This is the (updated) Cauchy stress
    //pstress_new[idx] = pstress[idx] + 
                       //(DPrime*2.*G + Identity*bulk*D.Trace())*delT;
      pstress_new[idx] = pstress[idx] - Identity*onethird*pstress[idx].Trace();
      pstress_new[idx](0,0) += (DPrime(0,0)*2.*G)*delT;
      pstress_new[idx](0,1) += (DPrime(0,1)*G)*delT;
      pstress_new[idx](0,2) += (DPrime(0,2)*G)*delT;
      pstress_new[idx](1,0) += (DPrime(1,0)*G)*delT;
      pstress_new[idx](1,1) += (DPrime(1,1)*2.*G)*delT;
      pstress_new[idx](1,2) += (DPrime(1,2)*G)*delT;
      pstress_new[idx](2,0) += (DPrime(2,0)*G)*delT;
      pstress_new[idx](2,1) += (DPrime(2,1)*G)*delT;
      pstress_new[idx](2,2) += (DPrime(2,2)*2.*G)*delT;

      // compute second invariant of deviatoric stress
      double aj2 = 0.5*(pstress_new[idx](0,0)*pstress_new[idx](0,0) +
                        pstress_new[idx](1,1)*pstress_new[idx](1,1) +
                        pstress_new[idx](2,2)*pstress_new[idx](2,2)) +
                        pstress_new[idx](0,1)*pstress_new[idx](0,1) +
                        pstress_new[idx](0,2)*pstress_new[idx](0,2) +
                        pstress_new[idx](1,2)*pstress_new[idx](1,2);
      double g0 = d_initialData.a0 + (d_initialData.a1 + d_initialData.a2*pres_guage)*pres_guage;
      if(g0<0.0) g0 = 0.0;

      double ratio;
      if(aj2<g0) ratio = 1.0;
      else ratio = sqrt(g0/(aj2+1.0e-10));

      //cout<<" aj2 "<<aj2<<" g0 "<<g0<<" ratio "<<ratio<<" pres_guage "<<pres_guage<<endl;

      pstress_new[idx] = pstress_new[idx]*ratio - Identity*pres_guage;

      // Compute the strain energy for all the particles
      Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

      double e = (D(0,0)*AvgStress(0,0) +
                  D(1,1)*AvgStress(1,1) +
                  D(2,2)*AvgStress(2,2) +
               2.*(D(0,1)*AvgStress(0,1) +
                   D(0,2)*AvgStress(0,2) +
                   D(1,2)*AvgStress(1,2))) * pvolume_deformed[idx]*delT;

      se += e;

      // Compute wave speed at each particle, store the maximum
      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(d_sharedState->adjustDelt(patch->getLevel(), delT_new)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);

    delete interpolator;
  }
}

void SoilFoam::carryForward(const PatchSubset* patches,
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
    /*ParticleVariable<double> sv_min_new, p_sv_min_new;
    constParticleVariable<double> sv_min, p_sv_min;
    old_dw->get(sv_min,                sv_minLabel,                    pset);
    old_dw->get(p_sv_min,              p_sv_minLabel,                  pset);
    new_dw->allocateAndPut(sv_min_new,   sv_minLabel_preReloc,         pset);
    new_dw->allocateAndPut(p_sv_min_new, p_sv_minLabel_preReloc,       pset);

    for(ParticleSubset::iterator iter = pset->begin();
                                 iter != pset->end(); iter++){
      particleIndex idx = *iter;
      sv_min_new[idx] = sv_min[idx];
      p_sv_min_new[idx] = p_sv_min[idx];
    }*/

    new_dw->put(delt_vartype(d_sharedState->adjustDelt(patch->getLevel(), 1.e10)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void SoilFoam::addInitialComputesAndRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(sv_minLabel,       matlset);
  task->computes(p_sv_minLabel,     matlset);
}

void SoilFoam::addComputesAndRequires(Task* task,
                                              const MPMMaterial* matl,
                                              const PatchSet* patches ) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  // Other constitutive model and input dependent computes and requires
  task->requires(Task::OldDW, sv_minLabel,         matlset, gnone);
  task->requires(Task::OldDW, p_sv_minLabel,       matlset, gnone);
  if(flag->d_with_ice)
    task->requires(Task::NewDW, csv_minLabel,        matlset, gnone);

  task->computes(sv_minLabel_preReloc,       matlset);
  task->computes(p_sv_minLabel_preReloc,     matlset);
}

void 
SoilFoam::addComputesAndRequires(Task* task,
                                   const MPMMaterial* matl,
                                   const PatchSet* patches,
                                   const bool ) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  // Other constitutive model and input dependent computes and requires
  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::OldDW, sv_minLabel,       matlset,gnone);
  task->requires(Task::OldDW, p_sv_minLabel,     matlset,gnone);
  if(flag->d_with_ice)
    task->requires(Task::NewDW, csv_minLabel,      matlset,gnone);

  task->computes(sv_minLabel_preReloc,       matlset);
  task->computes(p_sv_minLabel_preReloc,     matlset);
}

double SoilFoam::computeRhoMicroCM(double pressure,
                                    const double p_ref,
                                    const MPMMaterial* matl,
                                    const double maxvolstrain)
{
  double pres_guage = pressure - p_ref;
  //if(pres_guage<d_initialData.pc) pres_guage = d_initialData.pc;

  int i1 = 0, i;
  for(i=1; i<9; i++){
    if(d_initialData.eps[i+1]<d_initialData.eps[i])
      if(maxvolstrain<d_initialData.eps[i]) i1 = i;
  }
  double pressmax = d_initialData.p[i1] + slope[i1]*(maxvolstrain - d_initialData.eps[i1]);
  double vol_strain;
  if(pres_guage>pressmax){
     i1 = 0;
     for(i=1; i<9; i++){
       if(d_initialData.eps[i+1]<d_initialData.eps[i])
         if(pres_guage>d_initialData.p[i]) i1 = i;
     }
     vol_strain = (pres_guage - d_initialData.p[i1])/slope[i1] + d_initialData.eps[i1];
  }else{
     vol_strain = maxvolstrain + (pressmax - pres_guage)/d_initialData.bulk;
  }
  double rho_orig = matl->getInitialDensity();
  double rho_cur= rho_orig/exp(vol_strain);

  return rho_cur;
}

void SoilFoam::computePressEOSCM(double rho_cur,double& pressure,
                                         double p_ref,
                                         double& dp_drho, double& tmp,
                                         const MPMMaterial* matl,
                                         const double maxvolstrain)
{
  double pres_guage;
  int i1 = 0, i;
  for(i=1; i<9; i++){
    if(d_initialData.eps[i+1]<d_initialData.eps[i])
      if(maxvolstrain<d_initialData.eps[i]) i1 = i;
  }
  double pressmax = d_initialData.p[i1] + slope[i1]*(maxvolstrain - d_initialData.eps[i1]);

  double rho_orig = matl->getInitialDensity();
  double vol_strain = log(rho_orig/rho_cur);

  double slope1 = 0.0;

  if(vol_strain>=maxvolstrain){
     pres_guage = pressmax - d_initialData.bulk*(vol_strain - maxvolstrain);
     slope1 = d_initialData.bulk;
  }else{
     i1 = 0;
     for(i=1; i<9; i++){
        if(d_initialData.eps[i+1]<d_initialData.eps[i])
        if(vol_strain<d_initialData.eps[i]) i1 = i;
     }
     pres_guage = d_initialData.p[i1] + slope[i1]*(vol_strain - d_initialData.eps[i1]);
     slope1 = -slope[i1];
  }

  if(pres_guage<=d_initialData.pc){
    pres_guage = d_initialData.pc;
    //  dp_drho = 0.0;
    //tmp = (4.*d_initialData.G/3.)/rho_cur;
  }//else{
    dp_drho = slope1/rho_cur;
    tmp = (slope1 + 4.*d_initialData.G/3.)/rho_cur;  // speed of sound squared
    //}
  pressure = pres_guage + p_ref;
  //cout <<" load "<<vol_strain<<" "<<sv_min[idx]<<" "<<pres<<endl;
}

double SoilFoam::getCompressibility()
{
  cout << "NO VERSION OF getCompressibility EXISTS YET FOR SoilFoam"
       << endl;
  return 1.0;
}


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {

#if 0
static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(SoilFoam::CMData), sizeof(double)*3);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 3, 3, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(SoilFoam::CMData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other, "SoilFoam::CMData", true, &makeMPI_CMData);
   }
   return td;   
}
#endif

} // End namespace Uintah
