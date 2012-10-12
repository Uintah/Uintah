/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPM/ConstitutiveModel/HypoElasticImplicit.h>
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
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;

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
HypoElasticImplicit::computeStressTensorImplicit(const PatchSubset* patches,
                                                 const MPMMaterial* matl,
                                                 DataWarehouse* old_dw,
                                                 DataWarehouse* new_dw,
                                                 Solver* solver,
                                                 const bool )

{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
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
    constParticleVariable<Matrix3> psize;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress_new;
    constParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass;
    ParticleVariable<double> pvolume_deformed,pdTdt;
    constNCVariable<Vector> dispNew;
    
    DataWarehouse* parent_old_dw =
      new_dw->getOtherDataWarehouse(Task::ParentOldDW);
    pset = parent_old_dw->getParticleSubset(dwi, patch);
    parent_old_dw->get(px,                  lb->pXLabel,                  pset);
    parent_old_dw->get(psize,               lb->pSizeLabel,               pset);
    parent_old_dw->get(pmass,               lb->pMassLabel,               pset);
    parent_old_dw->get(pstress,             lb->pStressLabel,             pset);
    parent_old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(dispNew,lb->dispNewLabel,dwi,patch, Ghost::AroundCells,1);
  
    new_dw->allocateAndPut(pstress_new,      lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,  pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,   pset);
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
        pdTdt[idx] = 0.;
      }
    }
    else{
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;

        pdTdt[idx] = 0.;

        dispGrad.set(0.0);
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        for(int k = 0; k < 8; k++) {
          const Vector& disp = dispNew[ni[k]];
          for (int j = 0; j<3; j++){
            for (int i = 0; i<3; i++) {
              dispGrad(i,j) += disp[i] * d_S[k][j]* oodx[j];
            }
          }
        }

        int dof[24];
        loadBMats(l2g,dof,B,Bnl,d_S,ni,oodx);

        // Calculate the strain (here called D), and deviatoric rate DPrime
        Matrix3 e = (dispGrad + dispGrad.Transpose())*.5;
        Matrix3 ePrime = e - Identity*onethird*e.Trace();

        // This is the (updated) Cauchy stress

        pstress_new[idx] = pstress[idx] + (ePrime*2.*G+Identity*K*e.Trace());

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
}


void 
HypoElasticImplicit::computeStressTensorImplicit(const PatchSubset* patches,
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

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();

    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> psize;
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3> pstress_new;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<double> pvolume;
    ParticleVariable<double> pvolume_deformed,pdTdt;
    constNCVariable<Vector> dispNew;
    delt_vartype delT;

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

    new_dw->get(dispNew,lb->dispNewLabel,dwi,patch,Ghost::AroundCells,1);

    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    new_dw->allocateAndPut(pstress_new,      lb->pStressLabel_preReloc,  pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,   pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,    pset);
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
        pdTdt[idx] = 0.;
      }
    }
    else{
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
        
        pdTdt[idx] = 0.;

        dispGrad.set(0.0);
        // Get the node indices that surround the cell
        
        interpolator->findCellAndShapeDerivatives(px[idx], ni, d_S, psize[idx],deformationGradient[idx]);
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
      
      if (flag->d_reductionVars->accStrainEnergy ||
          flag->d_reductionVars->strainEnergy) {
        new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
      }
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
                                                 const bool /*recurse*/,
                                                 const bool SchedParent) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  bool reset = flag->d_doGridReset;

  addSharedCRForImplicitHypo(task, matlset, reset, true, SchedParent);
}

void HypoElasticImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  bool reset = flag->d_doGridReset;
                                                                                
  addSharedCRForImplicitHypo(task, matlset, reset);
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double HypoElasticImplicit::computeRhoMicroCM(double pressure, 
                                              const double p_ref,
                                              const MPMMaterial* matl,
                                              double temperature,
                                              double rho_guess)
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
                                            const MPMMaterial* matl, 
                                            double temperature)
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
