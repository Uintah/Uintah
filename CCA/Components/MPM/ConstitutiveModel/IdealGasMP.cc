#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/IdealGasMP.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> //for Fracture
#include <Packages/Uintah/Core/Grid/NodeIterator.h> 
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

#define FRACTURE
#undef FRACTURE

IdealGasMP::IdealGasMP(ProblemSpecP& ps,  MPMLabel* Mlb, int n8or27)
{
  lb = Mlb;

  ps->require("gamma", d_initialData.gamma);
  ps->require("specific_heat",d_initialData.cv);

  d_8or27 = n8or27;
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }

}

IdealGasMP::~IdealGasMP()
{
}

void IdealGasMP::initializeCMData(const Patch* patch,
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
  new_dw->allocateAndPut(pstress,            lb->pStressLabel,            pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
      deformationGradient[*iter] = Identity;
      pstress[*iter] = zero;
  }
  computeStableTimestep(patch, matl, new_dw);
}

void IdealGasMP::allocateCMDataAddRequires(Task* task,
						   const MPMMaterial* matl,
						   const PatchSet* patch,
						   MPMLabel* lb) const
{
  task->requires(Task::OldDW,lb->pDeformationMeasureLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pStressLabel, Ghost::None);
}

void IdealGasMP::allocateCMDataAdd(DataWarehouse* new_dw,
				   ParticleSubset* addset,
				   map<const VarLabel*, ParticleVariableBase*>* newState,
				   ParticleSubset* delset,
				   DataWarehouse* old_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3  zero(0.);
  ParticleSubset::iterator n,o;

  ParticleVariable<Matrix3> deformationGradient, pstress;

  constParticleVariable<Matrix3> o_DeformGrad, o_Stress;

  new_dw->allocateTemporary(deformationGradient,addset);
  new_dw->allocateTemporary(pstress,addset);

  old_dw->get(o_DeformGrad,lb->pDeformationMeasureLabel,delset);
  old_dw->get(o_Stress,lb->pStressLabel,delset);

  n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    deformationGradient[*n] = o_DeformGrad[*o];
    pstress[*n] = zero;
  }

  (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  (*newState)[lb->pStressLabel]=pstress.clone();
}


void IdealGasMP::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(lb->pDeformationMeasureLabel);
   from.push_back(lb->pStressLabel);

   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
}

void IdealGasMP::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume, ptemp;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,        pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,      pset);
  new_dw->get(ptemp,     lb->pTemperatureLabel, pset);
  new_dw->get(pvelocity, lb->pVelocityLabel,    pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double gamma = d_initialData.gamma;
  double cv    = d_initialData.cv;

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     double rhoM = pmass[idx]/pvolume[idx];
     double dp_drho = (gamma - 1.0)*cv*ptemp[idx];
     double dp_de   = (gamma - 1.0)*rhoM;

     double p = (gamma - 1.0)*rhoM*cv*ptemp[idx];

     double tmp = dp_drho + dp_de * p /(rhoM * rhoM);

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt(tmp);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void IdealGasMP::computeStressTensor(const PatchSubset* patches,
				      const MPMMaterial* matl,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    Matrix3 velGrad,deformationGradientInc;
    double J,p,se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Matrix3 Identity;

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass,ptemp;
    ParticleVariable<double> pvolume_deformed;
    constParticleVariable<Vector> pvelocity, psize;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;

    Ghost::GhostType  gac   = Ghost::AroundCells;

    old_dw->get(px,                          lb->pXLabel,                 pset);
    old_dw->get(pmass,                       lb->pMassLabel,              pset);
    old_dw->get(ptemp,                       lb->pTemperatureLabel,       pset);
    old_dw->get(pvelocity,                   lb->pVelocityLabel,          pset);
    old_dw->get(deformationGradient,         lb->pDeformationMeasureLabel,pset);
    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,    pset);
    if(d_8or27==27){
      old_dw->get(psize,                     lb->pSizeLabel,              pset);
    }
    new_dw->allocateAndPut(deformationGradient_new,
                                   lb->pDeformationMeasureLabel_preReloc, pset);

    new_dw->get(gvelocity, lb->gVelocityLabel, dwi,patch, gac, NGN);
    old_dw->get(delT, lb->delTLabel);

#ifdef FRACTURE
    constParticleVariable<Short27> pgCode;
    new_dw->get(pgCode, lb->pgCodeLabel, pset);
    constNCVariable<Vector> Gvelocity;
    new_dw->get(Gvelocity,lb->GVelocityLabel, dwi, patch, gac, NGN);
#endif
    
    double gamma = d_initialData.gamma;
    double cv    = d_initialData.cv;

    double rho_orig = matl->getInitialDensity();

    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
       particleIndex idx = *iter;

       // Get the node indices that surround the cell
       IntVector ni[MAX_BASIS];
       Vector d_S[MAX_BASIS];

       if(d_8or27==8){
          patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
        }
        else if(d_8or27==27){
          patch->findCellAndShapeDerivatives27(px[idx], ni, d_S,psize[idx]);
        }

        Vector gvel;
        velGrad.set(0.0);
        for(int k = 0; k < d_8or27; k++) {
#ifdef FRACTURE
	   if(pgCode[idx][k]==1) gvel = gvelocity[ni[k]];
	   if(pgCode[idx][k]==2) gvel = Gvelocity[ni[k]];
#else
	   gvel = gvelocity[ni[k]];
#endif
 	   for (int j = 0; j<3; j++){
	      for (int i = 0; i<3; i++) {
	         velGrad(i+1,j+1) += gvel[i] * d_S[k][j] * oodx[j];
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
      J    = deformationGradient_new[idx].Determinant();

      pvolume_deformed[idx]=(pmass[idx]/rho_orig)*J;
      double rhoM = pmass[idx]/pvolume_deformed[idx];
      double dp_drho = (gamma - 1.0)*cv*ptemp[idx];
      double dp_de   = (gamma - 1.0)*rhoM;

      p = (gamma - 1.0)*rhoM*cv*ptemp[idx];

      double tmp = dp_drho + dp_de * p /(rhoM * rhoM);

      pstress[idx] = Identity*(-p);

      Vector pvelocity_idx = pvelocity[idx];
      c_dil = sqrt(tmp);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
  		       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
		       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
    new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
  }
}

	 
void IdealGasMP::addComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const PatchSet*) const
{
   Ghost::GhostType  gac   = Ghost::AroundCells;
   const MaterialSubset* matlset = matl->thisMaterial();
   task->requires(Task::OldDW, lb->pXLabel,             matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pMassLabel,          matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pTemperatureLabel,   matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pVelocityLabel,      matlset, Ghost::None);
   task->requires(Task::OldDW, lb->pDeformationMeasureLabel,
						        matlset, Ghost::None);
   if(d_8or27==27){
     task->requires(Task::OldDW, lb->pSizeLabel,        matlset, Ghost::None);
   }
   task->requires(Task::NewDW,lb->gVelocityLabel,matlset,gac,NGN);
   task->requires(Task::OldDW, lb->delTLabel);
#ifdef FRACTURE
   task->requires(Task::NewDW, lb->pgCodeLabel,          matlset, Ghost::None);
   task->requires(Task::NewDW, lb->GVelocityLabel,       matlset, gac, NGN);
#endif

   task->computes(lb->pStressLabel_preReloc,             matlset);
   task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
   task->computes(lb->pVolumeDeformedLabel,              matlset);
}

void 
IdealGasMP::addComputesAndRequires(Task* ,
				   const MPMMaterial* ,
				   const PatchSet* ,
				   const bool ) const
{
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double IdealGasMP::computeRhoMicroCM(double press, 
                                      const double Temp,
                                      const MPMMaterial*)
{
  double gamma = d_initialData.gamma;
  double cv    = d_initialData.cv;

  return  press/((gamma - 1.0)*cv*Temp);
}

void IdealGasMP::computePressEOSCM(const double rhoM,double& pressure, 
                                               const double Temp,
                                               double& dp_drho, double& tmp,
                                                const MPMMaterial*)
{
  double gamma = d_initialData.gamma;
  double cv    = d_initialData.cv;

  pressure   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  double dp_de   = (gamma - 1.0)*rhoM;
  tmp = dp_drho + dp_de * pressure/(rhoM*rhoM);    // C^2
}

double IdealGasMP::getCompressibility()
{
  return 1.0/101325.;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {

#if 0
static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(IdealGasMP::StateData), sizeof(double)*0);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(IdealGasMP::StateData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
			       "IdealGasMP::StateData", true, &makeMPI_CMData);
   }
   return td;
}
#endif
} // End namespace Uintah
