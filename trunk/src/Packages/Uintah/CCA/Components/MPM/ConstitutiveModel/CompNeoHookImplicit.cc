#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompNeoHookImplicit.h>
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
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <TauProfilerForSCIRun.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

CompNeoHookImplicit::CompNeoHookImplicit(ProblemSpecP& ps,  MPMLabel* Mlb, 
					 MPMFlags* Mflag)
  : ConstitutiveModel(Mlb,Mflag), ImplicitCM(Mlb)
{
  d_useModifiedEOS = false;
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  ps->get("useModifiedEOS",d_useModifiedEOS); 

}

CompNeoHookImplicit::CompNeoHookImplicit(const CompNeoHookImplicit* cm)
{

  d_useModifiedEOS = cm->d_useModifiedEOS;
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
}

CompNeoHookImplicit::~CompNeoHookImplicit()
{
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
   ParticleVariable<Matrix3> deformationGradient, pstress, bElBar;
   ParticleVariable<double> pIntHeatRate;

   new_dw->allocateAndPut(deformationGradient,lb->pDeformationMeasureLabel,
                                                                        pset);
   new_dw->allocateAndPut(pstress,lb->pStressLabel,                     pset);
   new_dw->allocateAndPut(bElBar,lb->bElBarLabel,                       pset);
   new_dw->allocateAndPut(pIntHeatRate,lb->pInternalHeatRateLabel,      pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
          deformationGradient[*iter] = Identity;
          pstress[*iter] = zero;
          bElBar[*iter] = Identity;
          pIntHeatRate[*iter] = 0.0;
   }

}

void CompNeoHookImplicit::allocateCMDataAddRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* ,
                                                    MPMLabel* lb) const
{

  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW,lb->pDeformationMeasureLabel_preReloc, 
                 matlset, Ghost::None);
  task->requires(Task::NewDW,lb->pStressLabel_preReloc, 
                 matlset, Ghost::None);
  task->requires(Task::NewDW,bElBarLabel_preReloc, 
                 matlset, Ghost::None);

  task->requires(Task::NewDW,lb->pDeformationMeasureLabel_preReloc,
                 matlset,Ghost::None);
}


void CompNeoHookImplicit::allocateCMDataAdd(DataWarehouse* new_dw,
                                            ParticleSubset* addset,
                                            map<const VarLabel*, ParticleVariableBase*>* newState,
                                            ParticleSubset* delset,
                                            DataWarehouse* )
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  
  ParticleVariable<Matrix3> deformationGradient, pstress, bElBar;
  ParticleVariable<double> pIntHeatRate;
  constParticleVariable<Matrix3> o_deformationGradient, o_stress, o_bElBar;
  constParticleVariable<double> o_pIntHeatRate;
  
  new_dw->allocateTemporary(deformationGradient,addset);
  new_dw->allocateTemporary(pstress,            addset);
  new_dw->allocateTemporary(bElBar,             addset);
  new_dw->allocateTemporary(pIntHeatRate,       addset);
  
  new_dw->get(o_deformationGradient,lb->pDeformationMeasureLabel_preReloc,
                                                                 delset);
  new_dw->get(o_stress,lb->pStressLabel_preReloc,                delset);
  new_dw->get(o_bElBar,bElBarLabel_preReloc,                     delset);
  new_dw->get(o_pIntHeatRate,lb->pInternalHeatRateLabel_preReloc,delset);

  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    deformationGradient[*n] = o_deformationGradient[*o];
    bElBar[*n] = o_bElBar[*o];
    pstress[*n] = o_stress[*o];
    pIntHeatRate[*n] = o_pIntHeatRate[*o];
  }
  
  (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  (*newState)[lb->pStressLabel]=pstress.clone();
  (*newState)[lb->bElBarLabel]=bElBar.clone();
  (*newState)[lb->pInternalHeatRateLabel]=pIntHeatRate.clone();
}

void CompNeoHookImplicit::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
   from.push_back(lb->bElBarLabel);
   from.push_back(lb->pInternalHeatRateLabel);

   to.push_back(lb->bElBarLabel_preReloc);
   to.push_back(lb->pInternalHeatRateLabel_preReloc);
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
#ifdef HAVE_PETSC
                                         MPMPetscSolver* solver,
#else
                                         SimpleSolver* solver,
#endif
                                         const bool )

{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
//    cerr <<"Doing computeStressTensor on " << patch->getID()
//       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    solver->copyL2G(l2g,patch);

    Matrix3 Shear,deformationGradientInc,dispGrad,fbar;

    Matrix3 Identity;

    LinearInterpolator* interpolator = new LinearInterpolator(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());

    
    Identity.Identity();
    
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    
    int dwi = matl->getDWIndex();

    ParticleSubset* pset;
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new,bElBar_new;
    constParticleVariable<Matrix3> deformationGradient,bElBar_old;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pvolumeold;
    constParticleVariable<double> ptemperature;
    ParticleVariable<double> pvolume_deformed;
    constNCVariable<Vector> dispNew;
    delt_vartype delT;
    
    DataWarehouse* parent_old_dw = 
      new_dw->getOtherDataWarehouse(Task::ParentOldDW);
    pset = parent_old_dw->getParticleSubset(dwi, patch);
    parent_old_dw->get(px,             lb->pXLabel,                  pset);
    parent_old_dw->get(pvolumeold,     lb->pVolumeOldLabel,          pset);
    parent_old_dw->get(ptemperature,   lb->pTemperatureLabel,        pset);
    parent_old_dw->get(deformationGradient,
                                       lb->pDeformationMeasureLabel, pset);
    parent_old_dw->get(bElBar_old,     lb->bElBarLabel,              pset);

    old_dw->get(dispNew,lb->dispNewLabel,dwi,patch, Ghost::AroundCells,1);
  
    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,  pset);

    new_dw->allocateTemporary(deformationGradient_new,pset);
    new_dw->allocateTemporary(bElBar_new,             pset);

    double shear = d_initialData.Shear;
    double bulk  = d_initialData.Bulk;
    
    double B[6][24];
    double Bnl[3][24];
#ifdef HAVE_PETSC
    PetscScalar v[576];
#else
    double v[576];
#endif

    if(matl->getIsRigid()){
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pstress[idx] = Matrix3(0.0);
        bElBar_new[idx] = Identity;
        pvolume_deformed[idx] = pvolumeold[idx];
      }
    }
    else{
      int extraFlushesLeft = flag->d_extraSolverFlushes;
      int flushSpacing = 0;

      if ( extraFlushesLeft > 0 ) {
	flushSpacing = pset->numParticles() / flag->d_extraSolverFlushes;
      }

      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        dispGrad.set(0.0);
        // Get the node indices that surround the cell
        vector<IntVector> ni;
	ni.reserve(8);
        vector<Vector> d_S;
	d_S.reserve(8);

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
        // Find the stressTensor using the displacement gradient
      
        // Compute the deformation gradient increment using the dispGrad
      
        deformationGradientInc = dispGrad + Identity;

        // Update the deformation gradient tensor to its time n+1 value.
        deformationGradient_new[idx] = deformationGradientInc *
                                       deformationGradient[idx];
      
        // get the volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();

        fbar = deformationGradientInc * 
           pow(deformationGradientInc.Determinant(),-1./3.);

        bElBar_new[idx] = fbar*bElBar_old[idx]*fbar.Transpose();

        // Shear is equal to the shear modulus times dev(bElBar)
        double mubar = 1./3. * bElBar_new[idx].Trace()*shear;
        Matrix3 shrTrl = (bElBar_new[idx]*shear - Identity*mubar);

        // get the hydrostatic part of the stress
        double p = bulk*log(J)/J;

        // compute the total stress (volumetric + deviatoric)
        pstress[idx] = Identity*p + shrTrl/J;
        //cout << "p = " << p << " J = " << J << " tdev = " << shrTrl << endl;

        double coef1 = bulk;
        double coef2 = 2.*bulk*log(J);
        double D[6][6];

        D[0][0] = coef1 - coef2 + 2.*mubar*2./3. - 2./3.*(2.*shrTrl(0,0));
        D[0][1] = coef1 - 2.*mubar*1./3. - 2./3.*(shrTrl(0,0) + shrTrl(1,1));
        D[0][2] = coef1 -2.*mubar*1./3. - 2./3.*(shrTrl(0,0) + shrTrl(2,2));
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

        // Print out stuff
        /*
        cout.setf(ios::scientific,ios::floatfield);
        cout.precision(10);
        cout << "B = " << endl;
        for(int kk = 0; kk < 24; kk++) {
          for (int ll = 0; ll < 6; ++ll) {
            cout << B[ll][kk] << " " ;
          }
          cout << endl;
        }
        cout << "Bnl = " << endl;
        for(int kk = 0; kk < 24; kk++) {
          for (int ll = 0; ll < 3; ++ll) {
            cout << Bnl[ll][kk] << " " ;
          }
          cout << endl;
        }
        cout << "D = " << endl;
        for(int kk = 0; kk < 6; kk++) {
          for (int ll = 0; ll < 6; ++ll) {
            cout << D[ll][kk] << " " ;
          }
          cout << endl;
        }
        cout << "Kmat = " << endl;
        for(int kk = 0; kk < 24; kk++) {
          for (int ll = 0; ll < 24; ++ll) {
            cout << kmat[ll][kk] << " " ;
          }
          cout << endl;
        }
        cout << "Kgeo = " << endl;
        for(int kk = 0; kk < 24; kk++) {
          for (int ll = 0; ll < 24; ++ll) {
            cout << kgeo[ll][kk] << " " ;
          }
          cout << endl;
        }
        */
        double volold = pvolumeold[idx];
        double volnew = pvolumeold[idx]*J;

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

	flushSpacing--;

	if ( ( flushSpacing <= 0 ) && ( extraFlushesLeft > 0 ) ) {
	  flushSpacing = pset->numParticles() / flag->d_extraSolverFlushes;
	  extraFlushesLeft--;
	  solver->flushMatrix();
	}

      }  // end of loop over particles
      
      while ( extraFlushesLeft ) {
	extraFlushesLeft--;
	solver->flushMatrix();
      }

    }
    delete interpolator;
  }
  solver->flushMatrix();
}


void 
CompNeoHookImplicit::computeStressTensor(const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)


{
   for(int pp=0;pp<patches->size();pp++){
     const Patch* patch = patches->get(pp);
     Matrix3 Shear,deformationGradientInc,dispGrad,fbar;

     Matrix3 Identity;

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
     ParticleVariable<Matrix3> deformationGradient_new,bElBar_new;
     constParticleVariable<Matrix3> deformationGradient,bElBar_old;
     ParticleVariable<Matrix3> pstress;
     constParticleVariable<double> pvolumeold;
     constParticleVariable<double> ptemperature;
     ParticleVariable<double> pvolume_deformed;
     constNCVariable<Vector> dispNew;
     delt_vartype delT;

     old_dw->get(delT,lb->delTLabel, getLevel(patches));
     old_dw->get(px,                  lb->pXLabel,                  pset);
     old_dw->get(pvolumeold,          lb->pVolumeOldLabel,          pset);
     old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);

     new_dw->get(dispNew,lb->dispNewLabel,dwi,patch,Ghost::AroundCells,1);
     new_dw->allocateAndPut(pstress,         lb->pStressLabel_preReloc,   pset);
     new_dw->allocateAndPut(pvolume_deformed,lb->pVolumeDeformedLabel,    pset);
     old_dw->get(deformationGradient,        lb->pDeformationMeasureLabel,pset);
     old_dw->get(bElBar_old,                 lb->bElBarLabel,             pset);
     new_dw->allocateAndPut(deformationGradient_new,
                            lb->pDeformationMeasureLabel_preReloc, pset);
     new_dw->allocateAndPut(bElBar_new,lb->bElBarLabel_preReloc,   pset);

     ParticleVariable<double> pIntHeatRate;
     new_dw->allocateAndPut(pIntHeatRate, lb->pInternalHeatRateLabel_preReloc, 
                           pset);

     double shear = d_initialData.Shear;
     double bulk  = d_initialData.Bulk;

    if(matl->getIsRigid()){
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pstress[idx] = Matrix3(0.0);
        bElBar_new[idx] = Identity;
        deformationGradient_new[idx] = Identity;
        pvolume_deformed[idx] = pvolumeold[idx];
        pIntHeatRate[idx] = 0.;
      }
    }
    else{
     for(ParticleSubset::iterator iter = pset->begin();
                                  iter != pset->end(); iter++){
        particleIndex idx = *iter;

        pIntHeatRate[idx] = 0.;
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

        // Find the stressTensor using the displacement gradient

        // Compute the deformation gradient increment using the dispGrad

        deformationGradientInc = dispGrad + Identity;

        // Update the deformation gradient tensor to its time n+1 value.
       
        deformationGradient_new[idx] = deformationGradientInc *
                                       deformationGradient[idx];

        // get the volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();

        fbar = deformationGradientInc * 
         pow(deformationGradientInc.Determinant(),-1./3.);

        bElBar_new[idx] = fbar*bElBar_old[idx]*fbar.Transpose();

        // Shear is equal to the shear modulus times dev(bElBar)
        double mubar = 1./3. * bElBar_new[idx].Trace()*shear;
        Matrix3 shrTrl = (bElBar_new[idx]*shear - Identity*mubar);

        // get the hydrostatic part of the stress
        double p = bulk*log(J)/J;

        // compute the total stress (volumetric + deviatoric)
        pstress[idx] = Identity*p + shrTrl/J;
        //cout << "Last:p = " << p << " J = " << J << " tdev = " << shrTrl << endl;

        pvolume_deformed[idx] = pvolumeold[idx]*J;
      }
     }
    delete interpolator;
   }
}

void CompNeoHookImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet* ,
                                                 const bool ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // task->requires(Task::OldDW, lb->pXLabel,      matlset, Ghost::None);
  // new version uses ParentOldDW

  task->requires(Task::ParentOldDW, lb->pXLabel,         matlset,Ghost::None);
  task->requires(Task::ParentOldDW, lb->pVolumeOldLabel, matlset,Ghost::None);
  task->requires(Task::ParentOldDW, lb->pDeformationMeasureLabel,
                                                         matlset,Ghost::None);
  task->requires(Task::ParentOldDW,lb->bElBarLabel,      matlset,Ghost::None);
  task->requires(Task::ParentOldDW,lb->pTemperatureLabel,matlset,Ghost::None);
  task->requires(Task::OldDW,lb->dispNewLabel,matlset,Ghost::AroundCells,1);

  task->computes(lb->pStressLabel_preReloc,matlset);  
  task->computes(lb->pVolumeDeformedLabel, matlset);

}

void CompNeoHookImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->pXLabel,                 matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeOldLabel,         matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);
  task->requires(Task::OldDW, lb->bElBarLabel,             matlset,Ghost::None);
  task->requires(Task::NewDW, lb->dispNewLabel,matlset,Ghost::AroundCells,1);
  task->requires(Task::OldDW, lb->delTLabel);

  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->bElBarLabel_preReloc,              matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);
  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pInternalHeatRateLabel_preReloc,   matlset);
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

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

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
