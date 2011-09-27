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


#include <CCA/Components/MPM/ConstitutiveModel/CompNeoHookImplicit.h>
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
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <TauProfilerForSCIRun.h>

using std::cerr;
using namespace Uintah;

CompNeoHookImplicit::CompNeoHookImplicit(ProblemSpecP& ps,  MPMFlags* Mflag)
  : ConstitutiveModel(Mflag), ImplicitCM(), d_active(0)
{
  d_useModifiedEOS = false;
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  ps->get("active",d_active); 
  ps->get("useModifiedEOS",d_useModifiedEOS); 
  d_8or27=Mflag->d_8or27;
}

CompNeoHookImplicit::CompNeoHookImplicit(const CompNeoHookImplicit* cm)
  : ConstitutiveModel(cm), ImplicitCM(cm), d_active(0)
{

  d_useModifiedEOS = cm->d_useModifiedEOS;
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
}

CompNeoHookImplicit::~CompNeoHookImplicit()
{
}

void 
CompNeoHookImplicit::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","comp_neo_hook");
  }
  
  cm_ps->appendElement("bulk_modulus",d_initialData.Bulk);
  cm_ps->appendElement("shear_modulus",d_initialData.Shear);
  cm_ps->appendElement("useModifiedEOS",d_useModifiedEOS);
  cm_ps->appendElement("active",d_active);
}


CompNeoHookImplicit* CompNeoHookImplicit::clone()
{
  return scinew CompNeoHookImplicit(*this);
}

void CompNeoHookImplicit::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();

   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<Matrix3> deformationGradient, pstress;

   new_dw->allocateAndPut(deformationGradient,lb->pDeformationMeasureLabel,
                                                                        pset);
   new_dw->allocateAndPut(pstress,lb->pStressLabel,                     pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
          deformationGradient[*iter] = Identity;
          pstress[*iter] = zero;
   }
}

void CompNeoHookImplicit::allocateCMDataAddRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* ,
                                                    MPMLabel* lb) const
{

  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::NewDW,lb->pStressLabel_preReloc, matlset, gnone);
  task->requires(Task::NewDW,lb->pDeformationMeasureLabel_preReloc,
                                                        matlset, gnone);
}


void CompNeoHookImplicit::allocateCMDataAdd(DataWarehouse* new_dw,
                                            ParticleSubset* addset,
                                            map<const VarLabel*,
                                            ParticleVariableBase*>* newState,
                                            ParticleSubset* delset,
                                            DataWarehouse* )
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  
  ParticleVariable<Matrix3> deformationGradient, pstress;
  constParticleVariable<Matrix3> o_deformationGradient, o_stress;
  
  new_dw->allocateTemporary(deformationGradient,addset);
  new_dw->allocateTemporary(pstress,            addset);
  
  new_dw->get(o_deformationGradient,lb->pDeformationMeasureLabel_preReloc,
                                                                 delset);
  new_dw->get(o_stress,lb->pStressLabel_preReloc,                delset);

  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    deformationGradient[*n] = o_deformationGradient[*o];
    pstress[*n] = o_stress[*o];
  }
  
  (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  (*newState)[lb->pStressLabel]=pstress.clone();
}

void CompNeoHookImplicit::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
}

void CompNeoHookImplicit::computeStableTimestep(const Patch*,
                                           const MPMMaterial*,
                                           DataWarehouse*)
{
  // Not used for the implicit models.
}

void 
CompNeoHookImplicit::computeStressTensor(const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw,
                                         Solver* solver,
                                         const bool )

{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
//    cerr <<"Doing computeStressTensor on " << patch->getID()
//       <<"\t\t\t\t IMPM"<< "\n" << "\n";

   IntVector lowIndex,highIndex;
   if(d_8or27==8){
     lowIndex = patch->getNodeLowIndex();
     highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
   } else if(d_8or27==27){
     lowIndex = patch->getExtraNodeLowIndex();
     highIndex = patch->getExtraNodeHighIndex()+IntVector(1,1,1);
   }

    Array3<int> l2g(lowIndex,highIndex);
    solver->copyL2G(l2g,patch);

    Matrix3 Shear,deformationGradientInc;
    Matrix3 Identity;
    Identity.Identity();

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();

    ParticleSubset* pset;
    constParticleVariable<Point> px;
    constParticleVariable<Vector> psize;
    ParticleVariable<Matrix3> deformationGradient_new, pstress;
    constParticleVariable<Matrix3> deformationGradient;
    constParticleVariable<double> pvolumeold, pmass;
    ParticleVariable<double> pvolume_deformed, pdTdt;
    delt_vartype delT;
    
    DataWarehouse* parent_old_dw = 
      new_dw->getOtherDataWarehouse(Task::ParentOldDW);
    pset = parent_old_dw->getParticleSubset(dwi, patch);
    parent_old_dw->get(delT,           lb->delTLabel,   getLevel(patches));
    parent_old_dw->get(px,             lb->pXLabel,                  pset);
    parent_old_dw->get(pvolumeold,     lb->pVolumeLabel,             pset);
    parent_old_dw->get(pmass,          lb->pMassLabel,               pset);
    parent_old_dw->get(psize,          lb->pSizeLabel,               pset);
    parent_old_dw->get(deformationGradient,
                                       lb->pDeformationMeasureLabel, pset);

    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,  pset);

    new_dw->allocateTemporary(deformationGradient_new,pset);
    Ghost::GhostType  gac   = Ghost::AroundCells;

    double shear = d_initialData.Shear;
    double bulk  = d_initialData.Bulk;

    double rho_orig = matl->getInitialDensity();
    
    if(matl->getIsRigid()){
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pstress[idx] = Matrix3(0.0);
        pvolume_deformed[idx] = pvolumeold[idx];
      }
    }
    else{
      if(flag->d_doGridReset){
        constNCVariable<Vector> dispNew;
        old_dw->get(dispNew,lb->dispNewLabel,dwi,patch, gac, 1);
        computeDeformationGradientFromIncrementalDisplacement(
                                                      dispNew, pset, px,
                                                      deformationGradient,
                                                      deformationGradient_new,
                                                      dx, psize, interpolator);
      }
      else if(!flag->d_doGridReset){
        constNCVariable<Vector> gdisplacement;
        old_dw->get(gdisplacement, lb->gDisplacementLabel,dwi,patch,gac,1);
        computeDeformationGradientFromTotalDisplacement(gdisplacement,
                                                        pset, px,
                                                        deformationGradient_new,
                                                        deformationGradient,
                                                        dx, psize,interpolator);
      }

      double time = d_sharedState->getElapsedTime();

      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Assign zero internal heating by default - modify if necessary.
        pdTdt[idx] = 0.0;

        // get the volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();

        Matrix3 bElBar_new = deformationGradient_new[idx]
                           * deformationGradient_new[idx].Transpose()
                           * pow(J,-(2./3.));

        // Shear is equal to the shear modulus times dev(bElBar)
        double mubar = 1./3. * bElBar_new.Trace()*shear;
        Matrix3 shrTrl = (bElBar_new*shear - Identity*mubar);

        double active_stress = d_active*(time+delT);

        // get the hydrostatic part of the stress
        double p = bulk*log(J)/J + active_stress;

        // compute the total stress (volumetric + deviatoric)
        pstress[idx] = Identity*p + shrTrl/J;
        //cout << "p = " << p << " J = " << J << " tdev = " << shrTrl << endl;

        double coef1 = bulk;
        double coef2 = 2.*bulk*log(J);
        double D[6][6];

        D[0][0] = coef1 - coef2 + 2.*mubar*2./3. - 2./3.*(2.*shrTrl(0,0));
        D[0][1] = coef1 - 2.*mubar*1./3. - 2./3.*(shrTrl(0,0) + shrTrl(1,1));
        D[0][2] = coef1 - 2.*mubar*1./3. - 2./3.*(shrTrl(0,0) + shrTrl(2,2));
        D[0][3] =  - 2./3.*(shrTrl(0,1));
        D[0][4] =  - 2./3.*(shrTrl(0,2));
        D[0][5] =  - 2./3.*(shrTrl(1,2));
        D[1][1] = coef1 - coef2 + 2.*mubar*2./3. - 2./3.*(2.*shrTrl(1,1));
        D[1][2] = coef1 - 2.*mubar*1./3. - 2./3.*(shrTrl(1,1) + shrTrl(2,2));
        D[1][3] =  - 2./3.*(shrTrl(0,1));
        D[1][4] =  - 2./3.*(shrTrl(0,2));
        D[1][5] =  - 2./3.*(shrTrl(1,2));
        D[2][2] = coef1 - coef2 + 2.*mubar*2./3. - 2./3.*(2.*shrTrl(2,2));
        D[2][3] =  - 2./3.*(shrTrl(0,1));
        D[2][4] =  - 2./3.*(shrTrl(0,2));
        D[2][5] =  - 2./3.*(shrTrl(1,2));
        D[3][3] =  -.5*coef2 + mubar;
        D[3][4] = 0.;
        D[3][5] = 0.;
        D[4][4] =  -.5*coef2 + mubar;
        D[4][5] = 0.;
        D[5][5] =  -.5*coef2 + mubar;

        double sig[3][3];
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            sig[i][j]=pstress[idx](i,j);
          }
        }

        double volold = (pmass[idx]/rho_orig);
        double volnew = volold*J;
        int nDOF=3*d_8or27;

        if(d_8or27==8){
          double B[6][24];
          double Bnl[3][24];
          int dof[24];
          double v[576];
          double kmat[24][24];
          double kgeo[24][24];

          // Fill in the B and Bnl matrices and the dof vector
          interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S, psize[idx],deformationGradient[idx]);
          loadBMats(l2g,dof,B,Bnl,d_S,ni,oodx);
          // kmat = B.transpose()*D*B*volold
          BtDB(B,D,kmat);
          // kgeo = Bnl.transpose*sig*Bnl*volnew;
          BnltDBnl(Bnl,sig,kgeo);

          for (int I = 0; I < nDOF;I++){
            for (int J = 0; J < nDOF; J++){
              v[nDOF*I+J] = kmat[I][J]*volold + kgeo[I][J]*volnew;
            }
          }
          solver->fillMatrix(nDOF,dof,nDOF,dof,v);
        } else{
          double B[6][81];
          double Bnl[3][81];
          int dof[81];
          double v[6561];
          double kmat[81][81];
          double kgeo[81][81];

          // the code that computes kmat doesn't yet know that D is symmetric
          D[1][0] = D[0][1];
          D[2][0] = D[0][2];
          D[3][0] = D[0][3];
          D[4][0] = D[0][4];
          D[5][0] = D[0][5];
          D[1][1] = D[1][1];
          D[2][1] = D[1][2];
          D[3][1] = D[1][3];
          D[4][1] = D[1][4];
          D[1][2] = D[2][1];
          D[2][2] = D[2][2];
          D[3][2] = D[2][3];
          D[1][3] = D[3][1];
          D[2][3] = D[3][2];
          D[4][3] = D[3][4];

          // Fill in the B and Bnl matrices and the dof vector
          interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S, psize[idx],deformationGradient[idx]);
          loadBMatsGIMP(l2g,dof,B,Bnl,d_S,ni,oodx);
          // kmat = B.transpose()*D*B*volold
          BtDBGIMP(B,D,kmat);
          // kgeo = Bnl.transpose*sig*Bnl*volnew;
          BnltDBnlGIMP(Bnl,sig,kgeo);

          for (int I = 0; I < nDOF;I++){
            for (int J = 0; J < nDOF; J++){
              v[nDOF*I+J] = kmat[I][J]*volold + kgeo[I][J]*volnew;
            }
          }
          solver->fillMatrix(nDOF,dof,nDOF,dof,v);
        }  // endif 8or27

        pvolume_deformed[idx] = volnew;

      }  // end of loop over particles
    }
    delete interpolator;
  }
}


void 
CompNeoHookImplicit::computeStressTensor(const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)


{
   for(int pp=0;pp<patches->size();pp++){
     const Patch* patch = patches->get(pp);
     Matrix3 Shear,deformationGradientInc;

     Matrix3 Identity;
     Identity.Identity();

     Vector dx = patch->dCell();

     int dwi = matl->getDWIndex();
     ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
     constParticleVariable<Point> px;
     constParticleVariable<Vector> psize;
     ParticleVariable<Matrix3> deformationGradient_new;
     constParticleVariable<Matrix3> deformationGradient;
     ParticleVariable<Matrix3> pstress;
     constParticleVariable<double> pvolumeold, pmass;
     ParticleVariable<double> pvolume_deformed, pdTdt;
     delt_vartype delT;

     old_dw->get(delT,lb->delTLabel, getLevel(patches));
     old_dw->get(px,                  lb->pXLabel,                  pset);
     old_dw->get(psize,               lb->pSizeLabel,               pset);
     old_dw->get(pvolumeold,          lb->pVolumeLabel,             pset);
     old_dw->get(pmass,               lb->pMassLabel,               pset);
     old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

     new_dw->allocateAndPut(pstress,         lb->pStressLabel_preReloc,   pset);
     new_dw->allocateAndPut(pvolume_deformed,lb->pVolumeDeformedLabel,    pset);
     new_dw->allocateAndPut(pdTdt,           lb->pdTdtLabel_preReloc,     pset);
     new_dw->allocateAndPut(deformationGradient_new,
                                   lb->pDeformationMeasureLabel_preReloc, pset);

     double shear = d_initialData.Shear;
     double bulk  = d_initialData.Bulk;

     double rho_orig = matl->getInitialDensity();

    if(matl->getIsRigid()){
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pstress[idx] = Matrix3(0.0);
        deformationGradient_new[idx] = Identity;
        pvolume_deformed[idx] = pvolumeold[idx];
      }
    }
    else{
     ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
     Ghost::GhostType  gac   = Ghost::AroundCells;
     if(flag->d_doGridReset){
        constNCVariable<Vector> dispNew;
        new_dw->get(dispNew,lb->dispNewLabel,dwi,patch, gac, 1);
        computeDeformationGradientFromIncrementalDisplacement(
                                                      dispNew, pset, px,
                                                      deformationGradient,
                                                      deformationGradient_new,
                                                      dx, psize, interpolator);
     }
     else /*if(!flag->d_doGridReset)*/{
        constNCVariable<Vector> gdisplacement;
        new_dw->get(gdisplacement, lb->gDisplacementLabel,dwi,patch,gac,1);
        computeDeformationGradientFromTotalDisplacement(gdisplacement,pset, px,
                                                        deformationGradient_new,
                                                        deformationGradient,
                                                        dx, psize,interpolator);
     }

     double time = d_sharedState->getElapsedTime();

     for(ParticleSubset::iterator iter = pset->begin();
                                  iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Assign zero internal heating by default - modify if necessary.
        pdTdt[idx] = 0.0;

        // get the volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();
        
        Matrix3 bElBar_new = deformationGradient_new[idx]
                           * deformationGradient_new[idx].Transpose()
                           * pow(J,-(2./3.));

        // Shear is equal to the shear modulus times dev(bElBar)
        double mubar = 1./3. * bElBar_new.Trace()*shear;
        Matrix3 shrTrl = (bElBar_new*shear - Identity*mubar);

        double active_stress = d_active*(time+delT);
        // get the hydrostatic part of the stress
        double p = bulk*log(J)/J + active_stress;

        // compute the total stress (volumetric + deviatoric)
        pstress[idx] = Identity*p + shrTrl/J;

        double volold = (pmass[idx]/rho_orig);
        pvolume_deformed[idx] = volold*J;
      }
      delete interpolator;
    }
   }
}

void CompNeoHookImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet* ,
                                                 const bool /*recurse*/,
                                                 const bool SchedParent) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  bool reset = flag->d_doGridReset;

  addSharedCRForImplicit(task, matlset, reset, true, SchedParent);
}

void CompNeoHookImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  bool reset = flag->d_doGridReset;

  addSharedCRForImplicit(task, matlset, reset);
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double CompNeoHookImplicit::computeRhoMicroCM(double pressure, 
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
    rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));
  }
  return rho_cur;
}

void CompNeoHookImplicit::computePressEOSCM(const double rho_cur,double& pressure, 
                                    const double p_ref,
                                    double& dp_drho, double& tmp,
                                    const MPMMaterial* matl)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  if(d_useModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;           // MODIFIED EOS
    double n = bulk/p_ref;
    pressure = A*pow(rho_cur/rho_orig,n);
    dp_drho  = (bulk/rho_orig)*pow(rho_cur/rho_orig,n-1);
    tmp      = dp_drho;         // speed of sound squared
  } else {                      // STANDARD EOS            
    double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
    tmp        = bulk/rho_cur;  // speed of sound squared
  }
}

double CompNeoHookImplicit::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}


namespace Uintah {
  
#if 0
  static MPI_Datatype makeMPI_CMData()
  {
    ASSERTEQ(sizeof(CompNeoHookImplicit::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(CompNeoHookImplicit::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "CompNeoHookImplicit::StateData", true, 
                                  &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
