#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/TransIsoHyper.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h> //added this for stiffness
#include <Packages/Uintah/Core/Math/Short27.h> //for Fracture
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

// _________________transversely isotropic hyperelastic material [Jeff Weiss's]

TransIsoHyper::TransIsoHyper(ProblemSpecP& ps,  MPMLabel* Mlb, 
                             MPMFlags* Mflag)
  //______________________CONSTRUCTOR (READS INPUT, INITIALIZES SOME MODULI)
{
  lb = Mlb;
  flag = Mflag;
  d_useModifiedEOS = false;

  //______________________material properties
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("c1", d_initialData.c1);//Mooney Rivlin constant 1
  ps->require("c2", d_initialData.c2);//Mooney Rivlin constant 2
  ps->require("c3", d_initialData.c3);//scales exponential stresses
  ps->require("c4", d_initialData.c4);//controls uncrimping of fibers
  ps->require("c5", d_initialData.c5);//straightened fibers modulus
  ps->require("fiber_stretch", d_initialData.lambda_star);
  ps->require("direction_of_symm", d_initialData.a0);
  ps->require("failure_option",d_initialData.failure);//failure flag True/False
  ps->require("max_fiber_strain",d_initialData.crit_stretch);
  ps->require("max_matrix_strain",d_initialData.crit_shear);
  ps->get("useModifiedEOS",d_useModifiedEOS);//no negative pressure for solids

  //______________________interpolation
  d_8or27 = flag->d_8or27;
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }

  pStretchLabel = VarLabel::create("p.stretch",
     ParticleVariable<double>::getTypeDescription());
  pStretchLabel_preReloc = VarLabel::create("p.stretch+",
     ParticleVariable<double>::getTypeDescription());

  pFailureLabel = VarLabel::create("p.fail",
     ParticleVariable<double>::getTypeDescription());
  pFailureLabel_preReloc = VarLabel::create("p.fail+",
     ParticleVariable<double>::getTypeDescription());
}

TransIsoHyper::TransIsoHyper(const TransIsoHyper* cm)
{
  lb = cm->lb;
  flag = cm->flag;
  NGN = cm->NGN;
  d_8or27 = cm->d_8or27;

  d_useModifiedEOS = cm->d_useModifiedEOS ;

  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.c1 = cm->d_initialData.c1;
  d_initialData.c2 = cm->d_initialData.c2;
  d_initialData.c3 = cm->d_initialData.c3;
  d_initialData.c4 = cm->d_initialData.c4;
  d_initialData.c5 = cm->d_initialData.c5;
  d_initialData.lambda_star = cm->d_initialData.lambda_star;
  d_initialData.a0 = cm->d_initialData.a0;
  d_initialData.failure = cm->d_initialData.failure;
  d_initialData.crit_stretch = cm->d_initialData.crit_stretch;
  d_initialData.crit_shear = cm->d_initialData.crit_shear;
}

TransIsoHyper::~TransIsoHyper()
  // _______________________DESTRUCTOR
{
  VarLabel::destroy(pStretchLabel);
  VarLabel::destroy(pStretchLabel_preReloc);
  VarLabel::destroy(pFailureLabel);
  VarLabel::destroy(pFailureLabel_preReloc);
}

void TransIsoHyper::initializeCMData(const Patch* patch,
                                     const MPMMaterial* matl,
                                     DataWarehouse* new_dw)
  // _____________________STRESS FREE REFERENCE CONFIG
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 Identity, zero(0.);
  Identity.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  //ParticleVariable<Matrix3> deformationGradient, pstress;
  ParticleVariable<double> stretch,fail;//fail_label
  //new_dw->allocateAndPut(deformationGradient,lb->pDeformationMeasureLabel,pset);
  //new_dw->allocateAndPut(pstress,            lb->pStressLabel,            pset);
  new_dw->allocateAndPut(fail,               pFailureLabel,               pset);
  new_dw->allocateAndPut(stretch,            pStretchLabel,               pset);

  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++){
    //deformationGradient[*iter] = Identity;
    //pstress[*iter] = zero;
    fail[*iter] = 1.0 ;
    stretch[*iter] = 1.0;
  }
  computeStableTimestep(patch, matl, new_dw);
}


void TransIsoHyper::allocateCMDataAddRequires(Task* task,
                                              const MPMMaterial* matl ,
                                              const PatchSet* patches,
                                              MPMLabel* lb) const
  // _________________________________________STILL EXPERIMENTAL
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);
  // Add requires local to this model
  task->requires(Task::NewDW,pFailureLabel_preReloc,
                 matlset, Ghost::None);
  task->requires(Task::NewDW,pStretchLabel_preReloc,
                 matlset, Ghost::None);
}


void TransIsoHyper::allocateCMDataAdd(DataWarehouse* new_dw,
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
  //ParticleVariable<Matrix3> deformationGradient, pstress;
  //constParticleVariable<Matrix3> o_defGrad, o_stress;
  ParticleVariable<double> stretch,fail;
  constParticleVariable<double> o_stretch,o_fail;

  //new_dw->allocateTemporary(deformationGradient,addset);
  //new_dw->allocateTemporary(pstress,            addset);
  new_dw->allocateTemporary(stretch,            addset);
  new_dw->allocateTemporary(fail,               addset);

  //new_dw->get(o_defGrad,lb->pDeformationMeasureLabel_preReloc,   delset);
  //new_dw->get(o_stress, lb->pStressLabel_preReloc,               delset);
  new_dw->get(o_stretch,pStretchLabel_preReloc,                  delset);
  new_dw->get(o_fail,   pFailureLabel_preReloc,                  delset);

  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    //deformationGradient[*n] = o_defGrad[*o];
    //pstress[*n] = o_stress[*o];
    stretch[*n] = o_stretch[*o];
    fail[*n] = o_fail[*o];
  }
  //(*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  //(*newState)[lb->pStressLabel]=pstress.clone();
  (*newState)[pStretchLabel]=stretch.clone();
  (*newState)[pFailureLabel]=fail.clone();
}

void TransIsoHyper::addParticleState(std::vector<const VarLabel*>& from,
                                     std::vector<const VarLabel*>& to)
  //______________________________KEEPS TRACK OF THE PARTICLES AND THE RELATED VARIABLES
  //______________________________(EACH CM ADD ITS OWN STATE VARS)
  //______________________________AS PARTICLES MOVE FROM PATCH TO PATCH
{
  // Add the local particle state data for this constitutive model.
  from.push_back(lb->pFiberDirLabel);
  from.push_back(pStretchLabel);
  from.push_back(pFailureLabel);

  to.push_back(lb->pFiberDirLabel_preReloc);
  to.push_back(pStretchLabel_preReloc);
  to.push_back(pFailureLabel_preReloc);
}

void TransIsoHyper::computeStableTimestep(const Patch* patch,
                                          const MPMMaterial* matl,
                                          DataWarehouse* new_dw)
  //__________________________TIME STEP DEPENDS ON:
  //__________________________CELL SPACING, VEL OF PARTICLE, MATERIAL WAVE SPEED @ EACH PARTICLE
  //__________________________REDUCTION OVER ALL dT'S FROM EVERY PATCH PERFORMED
  //__________________________(USE THE SMALLEST dT)
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

  // __________________________________________Compute wave speed at each particle, store the maximum

  double Bulk = d_initialData.Bulk;
  double c1 = d_initialData.c1;

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
    particleIndex idx = *iter;

    // this is valid only for F=Identity
    c_dil = sqrt((Bulk+2./3.*c1)*pvolume[idx]/pmass[idx]);

    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(delT_new)), 
              lb->delTLabel);
}

Vector TransIsoHyper::getInitialFiberDir()
{
  return d_initialData.a0;
}

void TransIsoHyper::computeStressTensor(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
  //___________________________________COMPUTES THE STRESS ON ALL THE PARTICLES IN A GIVEN PATCH FOR A GIVEN MATERIAL
  //___________________________________CALLED ONCE PER TIME STEP
  //___________________________________CONTAINS A COPY OF computeStableTimestep
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    //
    Matrix3 velGrad,deformationGradientInc;
    double J,p;
    double U,W,se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Matrix3 Identity;
    Matrix3 rightCauchyGreentilde_new, leftCauchyGreentilde_new;
    Matrix3 pressure, deviatoric_stress, fiber_stress;
    double I1tilde,I2tilde,I4tilde,lambda_tilde;
    double dWdI4tilde, d2WdI4tilde2;
    double shear;
    Vector deformed_fiber_vector;

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());


    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    //double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass,pvolume;
    ParticleVariable<double> pvolume_deformed;
    ParticleVariable<double> stretch;
    ParticleVariable<double> fail;
    constParticleVariable<double> fail_old;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Vector> pfiberdir;
    ParticleVariable<Vector> pfiberdir_carry;
    constNCVariable<Vector> gvelocity;
    constParticleVariable<Vector> psize;
    delt_vartype delT;

    Ghost::GhostType  gac   = Ghost::AroundCells;
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(pfiberdir,           lb->pFiberDirLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(fail_old,            pFailureLabel,                pset);//fail
    old_dw->get(psize,               lb->pSizeLabel,               pset);

    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc,  pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,   pset);
    new_dw->allocateAndPut(pfiberdir_carry,  lb->pFiberDirLabel_preReloc,pset);
    new_dw->allocateAndPut(deformationGradient_new,
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(stretch,          pStretchLabel_preReloc,     pset);
    new_dw->get(gvelocity, lb->gVelocityLabel,dwi,patch,gac,NGN);
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    new_dw->allocateAndPut(fail,             pFailureLabel_preReloc,     pset);//fail
    ParticleVariable<double> pIntHeatRate;
    new_dw->allocateAndPut(pIntHeatRate, lb->pInternalHeatRateLabel_preReloc, 
                           pset);//internal heating rate
    //_____________________________________________material parameters
    double Bulk  = d_initialData.Bulk;
    double c1 = d_initialData.c1;
    double c2 = d_initialData.c2;
    double c3 = d_initialData.c3;
    double c4 = d_initialData.c4;
    double c5 = d_initialData.c5;
    double lambda_star = d_initialData.lambda_star;
    double c6 = c3*(exp(c4*(lambda_star-1.))-1.)-c5*lambda_star;//c6 = y-intercept
    double rho_orig = matl->getInitialDensity();
    double failure = d_initialData.failure;
    double crit_shear = d_initialData.crit_shear;
    double crit_stretch = d_initialData.crit_stretch;

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pIntHeatRate[idx] = 0.0;
      // Get the node indices that surround the cell
      interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx]);

      Vector gvel;
      velGrad.set(0.0);
      for(int k = 0; k < d_8or27; k++) {
        gvel = gvelocity[ni[k]];
        for (int j = 0; j<3; j++){
          double d_SXoodx = d_S[k][j] * oodx[j];
          for (int i = 0; i<3; i++) {
            velGrad(i,j) += gvel[i] * d_SXoodx;
          }
        }
      }
      
      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad * delT + Identity;

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
        deformationGradient[idx];

      // get the volumetric part of the deformation
      J = deformationGradient_new[idx].Determinant();

      // carry forward fiber direction
      pfiberdir_carry[idx] = pfiberdir[idx];
      deformed_fiber_vector =pfiberdir[idx]; // not actually deformed yet

      //_______________________UNCOUPLE DEVIATORIC AND DILATIONAL PARTS
      //_______________________Ftilde=J^(-1/3)*F
      //_______________________Fvol=J^1/3*Identity

      //_______________________right Cauchy Green (C) tilde and invariants
      rightCauchyGreentilde_new = deformationGradient_new[idx].Transpose()
        * deformationGradient_new[idx]*pow(J,-(2./3.));
      I1tilde = rightCauchyGreentilde_new.Trace();
      I2tilde = .5*(I1tilde*I1tilde -
                 (rightCauchyGreentilde_new*rightCauchyGreentilde_new).Trace());
      I4tilde = Dot(deformed_fiber_vector,
                    (rightCauchyGreentilde_new*deformed_fiber_vector));
      lambda_tilde = sqrt(I4tilde);
      double I4 = I4tilde*pow(J,(2./3.));// For diagnostics only
      stretch[idx] = sqrt(I4);
      deformed_fiber_vector = deformationGradient_new[idx]*deformed_fiber_vector
        *(1./lambda_tilde*pow(J,-(1./3.)));
      Matrix3 DY(deformed_fiber_vector,deformed_fiber_vector);

      //________________________________left Cauchy Green (B) tilde
      leftCauchyGreentilde_new = deformationGradient_new[idx]
        * deformationGradient_new[idx].Transpose()*pow(J,-(2./3.));

      //________________________________strain energy derivatives
      if (lambda_tilde < 1.)
        {dWdI4tilde = 0.;
        d2WdI4tilde2 = 0.;
        shear = 2.*c1+c2;
        }
      else
        if (lambda_tilde < lambda_star)
          {
            dWdI4tilde = 0.5*c3*(exp(c4*(lambda_tilde-1.))-1.)
              /lambda_tilde/lambda_tilde;
            d2WdI4tilde2 = 0.25*c3*(c4*exp(c4*(lambda_tilde-1.))
               -1./lambda_tilde*(exp(c4*(lambda_tilde-1.))-1.))
              /(lambda_tilde*lambda_tilde*lambda_tilde);

            shear = 2.*c1+c2+I4tilde*(4.*d2WdI4tilde2*lambda_tilde*lambda_tilde
                                      -2.*dWdI4tilde*lambda_tilde);
          }
        else
          {
            dWdI4tilde = 0.5*(c5+c6/lambda_tilde)/lambda_tilde;
            d2WdI4tilde2 = -0.25*c6
              /(lambda_tilde*lambda_tilde*lambda_tilde*lambda_tilde);
            shear = 2.*c1+c2+I4tilde*(4.*d2WdI4tilde2*lambda_tilde*lambda_tilde
                                      -2.*dWdI4tilde*lambda_tilde);
          }

      // Compute deformed volume and local wave speed
      double rho_cur = rho_orig/J;
      pvolume_deformed[idx]=pmass[idx]/rho_cur;
      c_dil = sqrt((Bulk+1./3.*shear)/rho_cur);

      // Compute bulk viscosity
      /*
      double qVisco = 0.0;
      if (flag->d_artificial_viscosity) {
        Matrix3 tensorD = (velGrad + velGrad.Transpose())*0.5;
        double Dkk = tensorD.Trace();
        double c_bulk = sqrt(Bulk/rho_cur);
        qVisco = artificialBulkViscosity(Dkk, c_bulk, rho_cur, dx_ave);
      }
      */
      //________________________________Failure and stress terms
      fail[idx] = 0.;
      if (failure == 1)
        {double matrix_failed = 0.;
        double fiber_failed = 0.;
        //________________________________Mooney Rivlin deviatoric term +failure of matrix
        Matrix3 RCG;
        RCG = deformationGradient_new[idx].Transpose()*deformationGradient_new[idx];
        double e1,e2,e3;//eigenvalues of C=symm.+pos.def.->Dis<=0
        double Q,R,Dis;
        double pi = 3.1415926535897932384;
        double I1 = RCG.Trace();
        double I2 = .5*(I1*I1 -(RCG*RCG).Trace());
        double I3 = RCG.Determinant();
        Q = (1./9.)*(3.*I2-pow(I1,2));
        R = (1./54.)*(-9.*I1*I2+27.*I3+2.*pow(I1,3));
        Dis = pow(Q,3)+pow(R,2);
        if (Dis <= 1.e-5 && Dis >= 0.)
          {if (R >= -1.e-5 && R<= 1.e-5)
            e1 = e2 = e3 = I1/3.;
          else
            {
              e1 = 2.*pow(R,1./3.)+I1/3.;
              e3 = -pow(R,1./3.)+I1/3.;
              if (e1 < e3) swap(e1,e3);
              e2=e3;
            }
          }
        else
          {double theta = acos(R/pow(-Q,3./2.));
          e1 = 2.*pow(-Q,1./2.)*cos(theta/3.)+I1/3.;
          e2 = 2.*pow(-Q,1./2.)*cos(theta/3.+2.*pi/3.)+I1/3.;
          e3 = 2.*pow(-Q,1./2.)*cos(theta/3.+4.*pi/3.)+I1/3.;
          if (e1 < e2) swap(e1,e2);
          if (e1 < e3) swap(e1,e3);
          if (e2 < e3) swap(e2,e3);
          };
        double max_shear_strain = (e1-e3)/2.;
        if (max_shear_strain > crit_shear || fail_old[idx]== 1.0 || fail_old[idx] == 3.0)
          {deviatoric_stress = Identity*0.;
          fail[idx] = 1.;
          matrix_failed = 1.;
          }
        else
         {deviatoric_stress = (leftCauchyGreentilde_new*(c1+c2*I1tilde)
               - leftCauchyGreentilde_new*leftCauchyGreentilde_new*c2
               - Identity*(1./3.)*(c1*I1tilde+2.*c2*I2tilde))*2./J;
          }
        //________________________________fiber stress term + failure of fibers
        if (stretch[idx] <= crit_stretch || fail_old[idx] == 2. || fail_old[idx] == 3.)
          {fiber_stress = Identity*0.;
          fail[idx] = 2.;
          fiber_failed =1.;
          }
        else
          {fiber_stress = (DY*dWdI4tilde*I4tilde
                           - Identity*(1./3.)*dWdI4tilde*I4tilde)*2./J;
          }
        if ( (matrix_failed + fiber_failed) == 2. || fail_old[idx] == 3.)
          fail[idx] = 3.;
        //________________________________hydrostatic pressure term
        if (fail[idx] == 1.0 ||fail[idx] == 3.0)
          pressure = Identity*0.;
        else
          {
            p = Bulk*log(J)/J; // p -= qVisco;
            if (p >= -1.e-5 && p <= 1.e-5)
              p = 0.;
            pressure = Identity*p;
          }
        //_______________________________Cauchy stress
        pstress[idx] = pressure + deviatoric_stress + fiber_stress;
        }
      else
        {
          deviatoric_stress = (leftCauchyGreentilde_new*(c1+c2*I1tilde)
               - leftCauchyGreentilde_new*leftCauchyGreentilde_new*c2
               - Identity*(1./3.)*(c1*I1tilde+2.*c2*I2tilde))*2./J;
          fiber_stress = (DY*dWdI4tilde*I4tilde
                          - Identity*(1./3.)*dWdI4tilde*I4tilde)*2./J;
          p = Bulk*log(J)/J; // p -= qVisco;
          if (p >= -1.e-5 && p <= 1.e-5)
            p = 0.;
          pressure = Identity*p;
          //Cauchy stress
          pstress[idx] = pressure + deviatoric_stress + fiber_stress;
        }
      //________________________________end stress

      // Compute the strain energy for all the particles
      U = .5*log(J)*log(J)*Bulk;
      if (lambda_tilde < lambda_star)
        W = c1*(I1tilde-3.)+c2*(I2tilde-3.)+(exp(c4*(lambda_tilde-1.)-1.))*c3;
      else
        W = c1*(I1tilde-3.)+c2*(I2tilde-3.)+c5*lambda_tilde+c6*log(lambda_tilde);

      double e = (U + W)*pvolume_deformed[idx]/J;

      se += e;

      Vector pvelocity_idx = pvelocity[idx];


      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(delT_new)),
                lb->delTLabel);
    new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
    delete interpolator;
  }
}


void TransIsoHyper::carryForward(const PatchSubset* patches,
                                 const MPMMaterial* matl,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
  //___________________________________________________________used with RigidMPM
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
    constParticleVariable<Vector> pfibdir;
    ParticleVariable<double> pstretch;
    constParticleVariable<Vector> pfail_old;
    ParticleVariable<double> pfail;

    ParticleVariable<Vector> pfibdir_new;
    old_dw->get(pfibdir,         lb->pFiberDirLabel,                   pset);
    old_dw->get(pfail_old,       pFailureLabel,                        pset);

    new_dw->allocateAndPut(pfibdir_new,      lb->pFiberDirLabel_preReloc, pset);
    new_dw->allocateAndPut(pstretch,         pStretchLabel_preReloc,      pset);
    new_dw->allocateAndPut(pfail,            pFailureLabel_preReloc,      pset);

    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;
      pfibdir_new[idx] = pfibdir[idx];
      pstretch[idx] = 1.0;
      pfail[idx] = 0.0;
    }
    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(1.e10)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void TransIsoHyper::addInitialComputesAndRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pFailureLabel,              matlset);
  task->computes(pStretchLabel,              matlset);
  task->computes(lb->pStressLabel_preReloc,  matlset);
  task->computes(lb->pVolumeDeformedLabel,   matlset);
}

void TransIsoHyper::addComputesAndRequires(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet* patches) const
  //___________TELLS THE SCHEDULER WHAT DATA
  //___________NEEDS TO BE AVAILABLE AT THE TIME computeStressTensor IS CALLED
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);

  // Other constitutive model and input dependent computes and requires
  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::OldDW, lb->pFiberDirLabel, matlset,gnone);
  task->requires(Task::OldDW, pFailureLabel,      matlset,gnone);

  task->computes(lb->pFiberDirLabel_preReloc, matlset);
  task->computes(pStretchLabel_preReloc,      matlset);
  task->computes(pFailureLabel_preReloc,      matlset);
}

void TransIsoHyper::addComputesAndRequires(Task* ,
                                           const MPMMaterial* ,
                                           const PatchSet* ,
                                           const bool ) const
  //_________________________________________here this one's empty
{
}


// The "CM" versions use the pressure-volume relationship of the CNH model
double TransIsoHyper::computeRhoMicroCM(double pressure, 
                                        const double p_ref,
                                        const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double Bulk = d_initialData.Bulk;
  
  double p_gauge = pressure - p_ref;
  double rho_cur;
 
  if(d_useModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;           // MODIFIED EOS
    double n = p_ref/Bulk;
    rho_cur = rho_orig*pow(pressure/A,n);
  } else {                      // STANDARD EOS
    rho_cur = rho_orig*(p_gauge/Bulk + sqrt((p_gauge/Bulk)*(p_gauge/Bulk) +1));
  }
  return rho_cur;
}

void TransIsoHyper::computePressEOSCM(const double rho_cur,double& pressure, 
                                      const double p_ref,
                                      double& dp_drho, double& tmp,
                                      const MPMMaterial* matl)
{
  double Bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  if(d_useModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;           // MODIFIED EOS
    double n = Bulk/p_ref;
    pressure = A*pow(rho_cur/rho_orig,n);
    dp_drho  = (Bulk/rho_orig)*pow(rho_cur/rho_orig,n-1);
    tmp      = dp_drho;         // speed of sound squared
  } else {                      // STANDARD EOS            
    double p_g = .5*Bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*Bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
    tmp        = Bulk/rho_cur;  // speed of sound squared
  }
}

double TransIsoHyper::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {
  
#if 0
  static MPI_Datatype makeMPI_CMData()
  {
    ASSERTEQ(sizeof(TransIsoHyper::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(TransIsoHyper::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "TransIsoHyper::StateData", true, &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah

