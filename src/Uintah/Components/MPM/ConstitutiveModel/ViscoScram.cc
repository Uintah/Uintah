
#include "ConstitutiveModelFactory.h"
#include "ViscoScram.h"
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <SCICore/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah::MPM;
using SCICore::Math::Min;
using SCICore::Math::Max;
using SCICore::Geometry::Vector;

ViscoScram::ViscoScram(ProblemSpecP& ps)
{
  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

  p_cmdata_label = scinew VarLabel("p.cmdata",
                                ParticleVariable<CMData>::getTypeDescription());
  p_cmdata_label_preReloc = scinew VarLabel("p.cmdata+",
                                ParticleVariable<CMData>::getTypeDescription());

  bElBarLabel = scinew VarLabel("p.bElBar",
                ParticleVariable<Matrix3>::getTypeDescription());
 
  bElBarLabel_preReloc = scinew VarLabel("p.bElBar+",
                ParticleVariable<Matrix3>::getTypeDescription());
}

ViscoScram::~ViscoScram()
{
  // Destructor

  delete p_cmdata_label;
  delete p_cmdata_label_preReloc;
  delete bElBarLabel;
  delete bElBarLabel_preReloc;
 
}

void ViscoScram::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouseP& new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();
   //   const MPMLabel* lb = MPMLabel::getLabels();

   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<CMData> cmdata;
   new_dw->allocate(cmdata, p_cmdata_label, pset);
   ParticleVariable<Matrix3> deformationGradient;
   new_dw->allocate(deformationGradient, lb->pDeformationMeasureLabel, pset);
   ParticleVariable<Matrix3> pstress;
   new_dw->allocate(pstress, lb->pStressLabel, pset);
   ParticleVariable<Matrix3> bElBar;
   new_dw->allocate(bElBar,  bElBarLabel, pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
          cmdata[*iter] = d_initialData;
          deformationGradient[*iter] = Identity;
          bElBar[*iter] = Identity;
          pstress[*iter] = zero;
   }
   new_dw->put(cmdata, p_cmdata_label);
   new_dw->put(deformationGradient, lb->pDeformationMeasureLabel);
   new_dw->put(pstress, lb->pStressLabel);
   new_dw->put(bElBar, bElBarLabel);

   computeStableTimestep(patch, matl, new_dw);

}

void ViscoScram::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(p_cmdata_label);
   from.push_back(bElBarLabel);
   to.push_back(p_cmdata_label_preReloc);
   to.push_back(bElBarLabel_preReloc);
}

void ViscoScram::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouseP& new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();
  //  const MPMLabel* lb = MPMLabel::getLabels();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<CMData> cmdata;
  new_dw->get(cmdata, p_cmdata_label, pset);
  ParticleVariable<double> pmass;
  new_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     double mu = cmdata[idx].Shear;
     double bulk = cmdata[idx].Bulk;
     c_dil = sqrt((bulk + 4.*mu/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void ViscoScram::computeStressTensor(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouseP& old_dw,
                                        DataWarehouseP& new_dw)
{
  Matrix3 velGrad,Shear,fbar,deformationGradientInc;
  double J,p,IEl,muBar,U,W,se=0.;
  double c_dil=0.0,Jinc;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  double onethird = (1.0/3.0);
  Matrix3 Identity;

  Identity.Identity();

  Vector dx = patch->dCell();
  double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

  int matlindex = matl->getDWIndex();
  //  const MPMLabel* lb = MPMLabel::getLabels();
  // Create array for the particle position
  ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<Point> px;
  old_dw->get(px, lb->pXLabel, pset);

  // Create array for the particle deformation
  ParticleVariable<Matrix3> deformationGradient;
  old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
  ParticleVariable<Matrix3> bElBar;
  old_dw->get(bElBar, bElBarLabel, pset);

  // Create array for the particle stress
  ParticleVariable<Matrix3> pstress;
  old_dw->get(pstress, lb->pStressLabel, pset);

  // Retrieve the array of constitutive parameters
  ParticleVariable<CMData> cmdata;
  old_dw->get(cmdata, p_cmdata_label, pset);
  ParticleVariable<double> pmass;
  old_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  old_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  old_dw->get(pvelocity, lb->pVelocityLabel, pset);

  NCVariable<Vector> gvelocity;

  new_dw->get(gvelocity, lb->gMomExedVelocityLabel, matlindex,patch,
            Ghost::AroundCells, 1);
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  for(ParticleSubset::iterator iter = pset->begin();
     iter != pset->end(); iter++){
     particleIndex idx = *iter;

     velGrad.set(0.0);
     // Get the node indices that surround the cell
     IntVector ni[8];
     Vector d_S[8];
     if(!patch->findCellAndShapeDerivatives(px[idx], ni, d_S))
         continue;

      for(int k = 0; k < 8; k++) {
          Vector& gvel = gvelocity[ni[k]];
          for (int j = 0; j<3; j++){
            for (int i = 0; i<3; i++) {
                velGrad(i+1,j+1)+=gvel(i) * d_S[k](j) * oodx[j];
            }
          }
      }

    // Calculate the stress Tensor (symmetric 3 x 3 Matrix) given the
    // time step and the velocity gradient and the material constants
    double shear = cmdata[idx].Shear;
    double bulk  = cmdata[idx].Bulk;

    // Compute the deformation gradient increment using the time_step
    // velocity gradient
    // F_n^np1 = dudx * dt + Identity
    deformationGradientInc = velGrad * delT + Identity;

    Jinc = deformationGradientInc.Determinant();

    // Update the deformation gradient tensor to its time n+1 value.
    deformationGradient[idx] = deformationGradientInc *
                             deformationGradient[idx];

    // get the volume preserving part of the deformation gradient increment
    fbar = deformationGradientInc * pow(Jinc,-onethird);

    bElBar[idx] = fbar*bElBar[idx]*fbar.Transpose();
    IEl = onethird*bElBar[idx].Trace();

    // Shear is equal to the shear modulus times dev(bElBar)
    Shear = (bElBar[idx] - Identity*IEl)*shear;

    // get the volumetric part of the deformation
    J = deformationGradient[idx].Determinant();

    // get the hydrostatic part of the stress
    p = 0.5*bulk*(J - 1.0/J);

    // compute the total stress (volumetric + deviatoric)
    pstress[idx] = Identity*p + Shear/J;

    // Compute the strain energy for all the particles
    U = .5*bulk*(.5*(pow(J,2.0) - 1.0) - log(J));
    W = .5*shear*(bElBar[idx].Trace() - 3.0);

    pvolume[idx]=Jinc*pvolume[idx];

    se += (U + W)*pvolume[idx]/J;

    // Compute wave speed at each particle, store the maximum
    muBar = IEl * shear;

    if(pmass[idx] > 0){
      c_dil = sqrt((bulk + 4.*shear/3.)*pvolume[idx]/pmass[idx]);
    }
    else{
      c_dil = 0.0;
      pvelocity[idx] = Vector(0.0,0.0,0.0);
    }
    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }

  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
  new_dw->put(pstress, lb->pStressLabel_preReloc);
  new_dw->put(deformationGradient, lb->pDeformationMeasureLabel_preReloc);
  new_dw->put(bElBar, bElBarLabel_preReloc);

  // Put the strain energy in the data warehouse
  new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);

  // This is just carried forward with the updated alpha
  new_dw->put(cmdata, p_cmdata_label_preReloc);
  // Volume is currently being carried forward, will be updated
  new_dw->put(pvolume,lb->pVolumeDeformedLabel);
}

double ViscoScram::computeStrainEnergy(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouseP& new_dw)
{
  double se=0;
#if 0
  double U,W,J,se=0;
  int matlindex = matl->getDWIndex();
  //  const MPMLabel* lb = MPMLabel::getLabels();
  // Create array for the particle deformation
  ParticleVariable<Matrix3> deformationGradient;
  new_dw->get(deformationGradient, lb->pDeformationMeasureLabel,
              matlindex, patch, Ghost::None, 0);

  // Get the elastic part of the shear strain
  ParticleVariable<Matrix3> bElBar;
  new_dw->get(bElBar, bElBarLabel, matlindex, patch, Ghost::None, 0);
  // Retrieve the array of constitutive parameters
  ParticleVariable<CMData> cmdata;
  new_dw->get(cmdata, p_cmdata_label, matlindex, patch, Ghost::None, 0);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume, lb->pVolumeLabel, matlindex, patch, Ghost::None, 0);

  ParticleSubset* pset = deformationGradient.getParticleSubset();
  ASSERT(pset == pvolume.getParticleSubset());

  for(ParticleSubset::iterator iter = pset->begin();
     iter != pset->end(); iter++){
     particleIndex idx = *iter;

     double shear = cmdata[idx].Shear;
     double bulk  = cmdata[idx].Bulk;

     J = deformationGradient[idx].Determinant();

     U = .5*bulk*(.5*(pow(J,2.0) - 1.0) - log(J));
     W = .5*shear*(bElBar[idx].Trace() - 3.0);

     se += (U + W)*pvolume[idx]/J;
  }
#endif

  return se;
}

void ViscoScram::addComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const Patch* patch,
					 DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw) const
{
  //  const MPMLabel* lb = MPMLabel::getLabels();
   task->requires(old_dw, lb->pXLabel, matl->getDWIndex(), patch,
                  Ghost::None);
   task->requires(old_dw, lb->pDeformationMeasureLabel, matl->getDWIndex(), patch,
                  Ghost::None);
   task->requires(old_dw, p_cmdata_label, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(old_dw, lb->pMassLabel, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(old_dw, lb->pVolumeLabel, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(new_dw, lb->gMomExedVelocityLabel, matl->getDWIndex(), patch,
                  Ghost::AroundCells, 1);
   task->requires(old_dw, bElBarLabel, matl->getDWIndex(), patch,
                  Ghost::None);
   task->requires(old_dw, lb->delTLabel);

   task->computes(new_dw, lb->pStressLabel_preReloc, matl->getDWIndex(),  patch);
   task->computes(new_dw, lb->pDeformationMeasureLabel_preReloc, matl->getDWIndex(), patch);
   task->computes(new_dw, bElBarLabel_preReloc, matl->getDWIndex(),  patch);
   task->computes(new_dw, p_cmdata_label_preReloc, matl->getDWIndex(),  patch);
   task->computes(new_dw, lb->pVolumeDeformedLabel, matl->getDWIndex(), patch);
}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {
   namespace MPM {

static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(ViscoScram::CMData), sizeof(double)*2);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 2, 2, MPI_DOUBLE, &mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(ViscoScram::CMData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
			       "ViscoScram::CMData", true, &makeMPI_CMData);
   }
   return td;
}
   }
}

// $Log$
// Revision 1.1  2000/08/21 18:37:41  guilkey
// Initial commit of ViscoScram stuff.  Don't get too excited yet,
// currently these are just cosmetically modified copies of CompNeoHook.
//
