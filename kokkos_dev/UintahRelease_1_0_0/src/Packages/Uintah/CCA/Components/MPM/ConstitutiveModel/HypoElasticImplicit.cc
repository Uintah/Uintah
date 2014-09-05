#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/HypoElasticImplicit.h>
#include <Packages/Uintah/Core/Grid/LinearInterpolator.h>
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
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

HypoElasticImplicit::HypoElasticImplicit(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag), ImplicitCM()
{
  ps->require("G",d_initialData.G);
  ps->require("K",d_initialData.K);
}

HypoElasticImplicit::HypoElasticImplicit(const HypoElasticImplicit* cm)
  : ConstitutiveModel(cm), ImplicitCM(cm)
{
  d_initialData.G = cm->d_initialData.G;
  d_initialData.K = cm->d_initialData.K;
}

HypoElasticImplicit::~HypoElasticImplicit()
{
}


void HypoElasticImplicit::outputProblemSpec(ProblemSpecP& ps,
                                            bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","hypo_elastic");
  }

  cm_ps->appendElement("G",d_initialData.G);
  cm_ps->appendElement("K",d_initialData.K);
}


HypoElasticImplicit* HypoElasticImplicit::clone()
{
  return scinew HypoElasticImplicit(*this);
}

void HypoElasticImplicit::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
  Matrix3 Identity, zero(0.);
  Identity.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<Matrix3> deformationGradient, pstress;
  new_dw->allocateAndPut(deformationGradient,lb->pDeformationMeasureLabel,pset);
  new_dw->allocateAndPut(pstress, lb->pStressLabel, pset);


  for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
     deformationGradient[*iter] = Identity;
     pstress[*iter] = zero;
  }
}

void
HypoElasticImplicit::allocateCMDataAddRequires( Task* task,
						const MPMMaterial* matl,
						const PatchSet* ,
						MPMLabel* lb ) const
{
  const MaterialSubset* matlset = matl->thisMaterial(); 
  task->requires(Task::NewDW,lb->pDeformationMeasureLabel_preReloc, 
                 matlset, Ghost::None);
  task->requires(Task::NewDW,lb->pStressLabel_preReloc, 
                 matlset, Ghost::None);
}

void
HypoElasticImplicit::allocateCMDataAdd( DataWarehouse* new_dw,
					ParticleSubset* addset,
					map<const VarLabel*, ParticleVariableBase*>* newState,
					ParticleSubset* delset,
					DataWarehouse* )
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  ParticleVariable<Matrix3> pstress,deformationGradient;
  constParticleVariable<Matrix3> o_stress, o_deformationGradient;

  new_dw->allocateTemporary(deformationGradient,addset);
  new_dw->allocateTemporary(pstress,            addset);

  new_dw->get(o_deformationGradient,lb->pDeformationMeasureLabel_preReloc,
                                                                        delset);
  new_dw->get(o_stress,             lb->pStressLabel_preReloc,          delset);

  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    deformationGradient[*n] = o_deformationGradient[*o];
    pstress[*n] = o_stress[*o];
  }

  (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  (*newState)[lb->pStressLabel]=pstress.clone();
}


void
HypoElasticImplicit::addParticleState( std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to )
{
}

void HypoElasticImplicit::computeStableTimestep(const Patch*,
                                           const MPMMaterial*,
                                           DataWarehouse*)
{
  // Not used in the implicit models
}

void 
HypoElasticImplicit::computeStressTensor(const PatchSubset* patches,
					 const MPMMaterial* matl,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw,
                                         Solver* solver,
					 const bool )

{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    LinearInterpolator* interpolator = new LinearInterpolator(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());

    IntVector lowIndex = patch->getInteriorNodeLowIndex();
    IntVector highIndex = patch->getInteriorNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    solver->copyL2G(l2g,patch);

    Matrix3 Shear,deformationGradientInc,dispGrad,fbar;
    double onethird = (1.0/3.0);
    
    Matrix3 Identity;
    
    Identity.Identity();
    
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    
    int dwi = matl->getDWIndex();

    ParticleSubset* pset;
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress_new;
    constParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass;
    ParticleVariable<double> pvolume_deformed;
    constNCVariable<Vector> dispNew;
    
    DataWarehouse* parent_old_dw =
      new_dw->getOtherDataWarehouse(Task::ParentOldDW);
    pset = parent_old_dw->getParticleSubset(dwi, patch);
    parent_old_dw->get(px,                  lb->pXLabel,                  pset);
    parent_old_dw->get(pmass,               lb->pMassLabel,               pset);
    parent_old_dw->get(pstress,             lb->pStressLabel,             pset);
    parent_old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(dispNew,lb->dispNewLabel,dwi,patch, Ghost::AroundCells,1);
  
    new_dw->allocateAndPut(pstress_new,      lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,  pset);
    new_dw->allocateTemporary(deformationGradient_new,pset);

    double G = d_initialData.G;
    double K  = d_initialData.K;

    double rho_orig = matl->getInitialDensity();

    double B[6][24];
    double Bnl[3][24];
    double v[576];

    if(matl->getIsRigid()){
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pstress_new[idx] = Matrix3(0.0);
        pvolume_deformed[idx] = pmass[idx]/rho_orig;
      }
    }
    else{
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;


        dispGrad.set(0.0);
        // Get the node indices that surround the cell

        interpolator->findCellAndShapeDerivatives(px[idx], ni, d_S);
        int dof[24];
        int l2g_node_num;
        for(int k = 0; k < 8; k++) {
          // Need to loop over the neighboring patches l2g to get the right
          // dof number.
          l2g_node_num = l2g[ni[k]];
          dof[3*k]  =l2g_node_num;
          dof[3*k+1]=l2g_node_num+1;
          dof[3*k+2]=l2g_node_num+2;

          const Vector& disp = dispNew[ni[k]];
	
       	  for (int j = 0; j<3; j++){
            for (int i = 0; i<3; i++) {
              dispGrad(i,j) += disp[i] * d_S[k][j]* oodx[j];
            }
          }

          B[0][3*k] = d_S[k][0]*oodx[0];
          B[3][3*k] = d_S[k][1]*oodx[1];
          B[5][3*k] = d_S[k][2]*oodx[2];
          B[1][3*k] = 0.;
          B[2][3*k] = 0.;
          B[4][3*k] = 0.;

          B[1][3*k+1] = d_S[k][1]*oodx[1];
          B[3][3*k+1] = d_S[k][0]*oodx[0];
          B[4][3*k+1] = d_S[k][2]*oodx[2];
          B[0][3*k+1] = 0.;
          B[2][3*k+1] = 0.;
          B[5][3*k+1] = 0.;

          B[2][3*k+2] = d_S[k][2]*oodx[2];
          B[4][3*k+2] = d_S[k][1]*oodx[1];
          B[5][3*k+2] = d_S[k][0]*oodx[0];
          B[0][3*k+2] = 0.;
          B[1][3*k+2] = 0.;
          B[3][3*k+2] = 0.;

          Bnl[0][3*k] = d_S[k][0]*oodx[0];
          Bnl[1][3*k] = 0.;
          Bnl[2][3*k] = 0.;
          Bnl[0][3*k+1] = 0.;
          Bnl[1][3*k+1] = d_S[k][1]*oodx[1];
          Bnl[2][3*k+1] = 0.;
          Bnl[0][3*k+2] = 0.;
          Bnl[1][3*k+2] = 0.;
          Bnl[2][3*k+2] = d_S[k][2]*oodx[2];
        }
      
        // Calculate the strain (here called D), and deviatoric rate DPrime
        Matrix3 e = (dispGrad + dispGrad.Transpose())*.5;
        Matrix3 ePrime = e - Identity*onethird*e.Trace();

        // This is the (updated) Cauchy stress

        pstress_new[idx] = pstress[idx] + (ePrime*2.*G+Identity*K*e.Trace());

//        cout << pstress_new[idx] << endl;

        // Compute the deformation gradient increment using the dispGrad
      
        deformationGradientInc = dispGrad + Identity;

        // Update the deformation gradient tensor to its time n+1 value.
        deformationGradient_new[idx] = deformationGradientInc *
                                       deformationGradient[idx];

        // get the volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();

        double E = 9.*K*G/(3.*K+G);
        double PR = (3.*K-E)/(6.*K);
        double C11 = E*(1.-PR)/((1.+PR)*(1.-2.*PR));
        double C12 = E*PR/((1.+PR)*(1.-2.*PR));
        double C44 = G;

        double D[6][6];
      
        D[0][0] = C11;
        D[0][1] = C12;
        D[0][2] = C12;
        D[0][3] = 0.;
        D[0][4] = 0.;
        D[0][5] = 0.;
        D[1][1] = C11;
        D[1][2] = C12;
        D[1][3] = 0.;
        D[1][4] = 0.;
        D[1][5] = 0.;
        D[2][2] = C11;
        D[2][3] = 0.;
        D[2][4] = 0.;
        D[2][5] = 0.;
        D[3][3] = C44;
        D[3][4] = 0.;
        D[3][5] = 0.;
        D[4][4] = C44;
        D[4][5] = 0.;
        D[5][5] = C44;
      
        D[1][0]=D[0][1];
        D[2][0]=D[0][2];
        D[2][1]=D[1][2];
        D[3][0]=D[0][3];
        D[3][1]=D[1][3];
        D[3][2]=D[2][3];
        D[4][0]=D[0][4];
        D[4][1]=D[1][4];
        D[4][2]=D[2][4];
        D[4][3]=D[3][4];
        D[5][0]=D[0][5];
        D[5][1]=D[1][5];
        D[5][2]=D[2][5];
        D[5][3]=D[3][5];
        D[5][4]=D[4][5];
      
        // kmat = B.transpose()*D*B*volold
        double kmat[24][24];
        BtDB(B,D,kmat);
        // kgeo = Bnl.transpose*sig*Bnl*volnew;
        double sig[3][3];
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            sig[i][j]=pstress[idx](i,j);
          }
        }
        double kgeo[24][24];
        BnltDBnl(Bnl,sig,kgeo);

        double volold = (pmass[idx]/rho_orig);
        double volnew = volold*J;

        pvolume_deformed[idx] = volnew;

        for(int ii = 0;ii<24;ii++){
          for(int jj = 0;jj<24;jj++){
            kmat[ii][jj]*=volold;
            kgeo[ii][jj]*=volnew;
          }
        }

	for (int I = 0; I < 24;I++){
	  for (int J = 0; J < 24; J++){
	    v[24*I+J] = kmat[I][J] + kgeo[I][J];
	  }
	}
        solver->fillMatrix(24,dof,24,dof,v);
     }
    }
    delete interpolator;
  }
  solver->flushMatrix();
}


void 
HypoElasticImplicit::computeStressTensor(const PatchSubset* patches,
					 const MPMMaterial* matl,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw)


{
   for(int pp=0;pp<patches->size();pp++){
    double se = 0.0;
    const Patch* patch = patches->get(pp);
    Matrix3 dispGrad,deformationGradientInc,Identity,zero(0.),One(1.);
    double Jinc;
    double onethird = (1.0/3.0);

    LinearInterpolator* interpolator = new LinearInterpolator(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());


    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();

    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3> pstress_new;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<double> pvolume;
    ParticleVariable<double> pvolume_deformed;
    constParticleVariable<Vector> pvelocity;
    constNCVariable<Vector> dispNew;
    delt_vartype delT;

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

    new_dw->get(dispNew,lb->dispNewLabel,dwi,patch,Ghost::AroundCells,1);

    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    new_dw->allocateAndPut(pstress_new,      lb->pStressLabel_preReloc,  pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,   pset);
    new_dw->allocateAndPut(deformationGradient_new,
			   lb->pDeformationMeasureLabel_preReloc,        pset);
 
    double G    = d_initialData.G;
    double bulk = d_initialData.K;

    if(matl->getIsRigid()){
      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pstress_new[idx] = Matrix3(0.0);
        deformationGradient_new[idx] = Identity;
        pvolume_deformed[idx] = pvolume[idx];
      }
    }
    else{
      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	particleIndex idx = *iter;
	
        dispGrad.set(0.0);
	// Get the node indices that surround the cell
	
	interpolator->findCellAndShapeDerivatives(px[idx], ni, d_S);
	for(int k = 0; k < 8; k++) {
	  const Vector& disp = dispNew[ni[k]];
	  
	  for (int j = 0; j<3; j++){
	    for (int i = 0; i<3; i++) {
	      dispGrad(i,j) += disp[i] * d_S[k][j]* oodx[j];
	    }
	  }
	}

      // Calculate the strain (here called D), and deviatoric rate DPrime
      Matrix3 D = (dispGrad + dispGrad.Transpose())*.5;
      Matrix3 DPrime = D - Identity*onethird*D.Trace();

      // This is the (updated) Cauchy stress

      pstress_new[idx] = pstress[idx] + (DPrime*2.*G + Identity*bulk*D.Trace());

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = dispGrad + Identity;

      Jinc = deformationGradientInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
                                     deformationGradient[idx];

      pvolume_deformed[idx]=Jinc*pvolume[idx];

      // Compute the strain energy for all the particles
      Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

      double e = (D(0,0)*AvgStress(0,0) +
	          D(1,1)*AvgStress(1,1) +
	          D(2,2)*AvgStress(2,2) +
              2.*(D(0,1)*AvgStress(0,1) +
                  D(0,2)*AvgStress(0,2) +
                  D(1,2)*AvgStress(1,2))) * pvolume_deformed[idx]*delT;
      
      se += e;
      }
      new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
    }
    delete interpolator;
   }
}

void HypoElasticImplicit::addInitialComputesAndRequires(Task*,
                                                const MPMMaterial*,
                                                const PatchSet*) const
{

}

void HypoElasticImplicit::addComputesAndRequires(Task* task,
						 const MPMMaterial* matl,
						 const PatchSet* ,
						 const bool ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->requires(Task::ParentOldDW, lb->pXLabel,         matlset,Ghost::None);
  task->requires(Task::ParentOldDW, lb->pMassLabel,      matlset,Ghost::None);
  task->requires(Task::ParentOldDW, lb->pDeformationMeasureLabel,
                                                         matlset,Ghost::None);
  task->requires(Task::OldDW,lb->dispNewLabel,matlset,Ghost::AroundCells,1);

  task->computes(lb->pStressLabel_preReloc,matlset);  
  task->computes(lb->pVolumeDeformedLabel, matlset);
}

void HypoElasticImplicit::addComputesAndRequires(Task* task,
						 const MPMMaterial* matl,
						 const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pXLabel,                 matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pStressLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVelocityLabel,          matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);
  task->requires(Task::NewDW, lb->dispNewLabel,   matlset,Ghost::AroundCells,1);

  task->computes(lb->pStressLabel_preReloc,                matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc,    matlset);
  task->computes(lb->pVolumeDeformedLabel,                 matlset);
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double HypoElasticImplicit::computeRhoMicroCM(double pressure, 
                                              const double p_ref,
                                              const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double bulk = d_initialData.K;

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;
}

void HypoElasticImplicit::computePressEOSCM(const double rho_cur,
                                            double& pressure, 
                                            const double p_ref,
                                            double& dp_drho, double& tmp,
                                            const MPMMaterial* matl)
{
  double bulk = d_initialData.K;
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = bulk/rho_cur;
}

double HypoElasticImplicit::getCompressibility()
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
    ASSERTEQ(sizeof(HypoElasticImplicit::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(HypoElasticImplicit::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "HypoElasticImplicit::StateData", true, 
				  &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
