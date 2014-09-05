#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/TransIsoHyperImplicit.h>
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
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <TauProfilerForSCIRun.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

// ____________________this is a transversely isotropic hyperelastic material [JW]
//_____________________see Material 18 in LSDYNA manual
//_____________________implicit MPM

TransIsoHyperImplicit::TransIsoHyperImplicit(ProblemSpecP& ps,  MPMLabel* Mlb, int n8or27)
//______________________CONSTRUCTOR (READS INPUT, INITIALIZES SOME MODULI)
{
  lb = Mlb;
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
  ps->get("useModifiedEOS",d_useModifiedEOS);//no negative pressure for solids

//______________________interpolation
  d_8or27 = n8or27;
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }

  pStretchLabel = VarLabel::create("p.stretch",
        ParticleVariable<double>::getTypeDescription());

  pStretchLabel_preReloc = VarLabel::create("p.stretch+",
        ParticleVariable<double>::getTypeDescription());

}

TransIsoHyperImplicit::TransIsoHyperImplicit(const TransIsoHyperImplicit* cm)
{
  lb = cm->lb;
  d_8or27 = cm->d_8or27;
  NGN = cm->NGN;

  d_useModifiedEOS = cm->d_useModifiedEOS ;

  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.c1 = cm->d_initialData.c1;
  d_initialData.c2 = cm->d_initialData.c2;
  d_initialData.c3 = cm->d_initialData.c3;
  d_initialData.c4 = cm->d_initialData.c4;
  d_initialData.c5 = cm->d_initialData.c5;
  d_initialData.lambda_star = cm->d_initialData.lambda_star;
  d_initialData.a0 = cm->d_initialData.a0;
}

TransIsoHyperImplicit::~TransIsoHyperImplicit()
// _______________________DESTRUCTOR
{
  VarLabel::destroy(pStretchLabel);
  VarLabel::destroy(pStretchLabel_preReloc);
}

void TransIsoHyperImplicit::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
// _____________________STRESS FREE REFERENCE CONFIG,
//______________________PLUS ESTIMATES TIME STEP THROUGHcomputeStableTimestep
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();

   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<Matrix3> deformationGradient, pstress;

   new_dw->allocateAndPut(deformationGradient,lb->pDeformationMeasureLabel,pset);
   new_dw->allocateAndPut(pstress,lb->pStressLabel,pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
          deformationGradient[*iter] = Identity;
          pstress[*iter] = zero;
   }

}

void TransIsoHyperImplicit::allocateCMDataAddRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* patch,
                                                    MPMLabel* lb) const
{

  //const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,lb->pDeformationMeasureLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pStressLabel, Ghost::None);
}


void TransIsoHyperImplicit::allocateCMDataAdd(DataWarehouse* new_dw,
                                            ParticleSubset* addset,
                                            map<const VarLabel*, ParticleVariableBase*>* newState,
                                            ParticleSubset* delset,
                                            DataWarehouse* old_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3  zero(0.);
  
  ParticleVariable<Matrix3> deformationGradient, pstress;
  constParticleVariable<Matrix3> o_deformationGradient, o_stress;
  
  new_dw->allocateTemporary(deformationGradient,addset);
  new_dw->allocateTemporary(pstress,addset);
  
  old_dw->get(o_deformationGradient,lb->pDeformationMeasureLabel,delset);
  old_dw->get(o_stress,lb->pStressLabel,delset);


  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    deformationGradient[*n] = o_deformationGradient[*o];
    pstress[*n] = zero;
  }
  
  (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  (*newState)[lb->pStressLabel]=pstress.clone();

}

void TransIsoHyperImplicit::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
   from.push_back(lb->pDeformationMeasureLabel);
   from.push_back(lb->pStressLabel);
   from.push_back(lb->pFiberDirLabel);
   from.push_back(pStretchLabel);

   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
   to.push_back(lb->pFiberDirLabel_preReloc);
   to.push_back(pStretchLabel_preReloc);
}

void TransIsoHyperImplicit::computeStableTimestep(const Patch*,
                                                  const MPMMaterial*,
                                                  DataWarehouse*)
{
  // Not used for the implicit models.
}

Vector TransIsoHyperImplicit::getInitialFiberDir()
{
  return d_initialData.a0;
}
//
void
TransIsoHyperImplicit::computeStressTensor(const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw,
#ifdef HAVE_PETSC
                                         MPMPetscSolver* solver,
#else
                                         SimpleSolver* solver,
#endif
                                         const bool recursion)
//___________________________________COMPUTES THE STRESS ON ALL THE PARTICLES IN A GIVEN PATCH FOR A GIVEN MATERIAL
//___________________________________CALLED ONCE PER TIME STEP
//___________________________________CONTAINS A COPY OF computeStableTimestep
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
//
    double p;
    Matrix3 rightCauchyGreentilde_new, leftCauchyGreentilde_new;
    double I1tilde,I2tilde,I4tilde,lambda_tilde;
    double dWdI4tilde, d2WdI4tilde2;
    double shear;
    Vector deformed_fiber_vector;
//
    Matrix3 Identity;
    
    Identity.Identity();
    
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    
    int dwi = matl->getDWIndex();

    ParticleSubset* pset;
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pvolumeold;
    constParticleVariable<double> ptemperature;
    ParticleVariable<double> pvolume_deformed;
// for LV appl.
    ParticleVariable<double> stretch;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Vector> pfiberdir;
//
    constNCVariable<Vector> dispNew;
    delt_vartype delT;
    
    DataWarehouse* parent_old_dw = 
      new_dw->getOtherDataWarehouse(Task::ParentOldDW);

    pset = parent_old_dw->getParticleSubset(dwi, patch);
    parent_old_dw->get(px,                 lb->pXLabel,                  pset);
    parent_old_dw->get(pvolumeold,         lb->pVolumeOldLabel,          pset);
    parent_old_dw->get(ptemperature,       lb->pTemperatureLabel,        pset);
    parent_old_dw->get(deformationGradient,lb->pDeformationMeasureLabel, pset);
    parent_old_dw->get(pfiberdir,          lb->pFiberDirLabel,           pset);

    old_dw->get(dispNew,lb->dispNewLabel,dwi,patch, Ghost::AroundCells,1);

    new_dw->allocateAndPut(pstress,         lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pvolume_deformed,lb->pVolumeDeformedLabel,  pset);
    new_dw->allocateTemporary(deformationGradient_new,pset);

//_____________________________________________material parameters
    double Bulk  = d_initialData.Bulk;
    double c1 = d_initialData.c1;
    double c2 = d_initialData.c2;
    double c3 = d_initialData.c3;
    double c4 = d_initialData.c4;
    double c5 = d_initialData.c5;
    double lambda_star = d_initialData.lambda_star;
    double c6 = c3*(exp(c4*(lambda_star-1.))-1.)-c5*lambda_star;//c6 = y-intercept
    
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
        pvolume_deformed[idx] = pvolumeold[idx];
      }
    }
    else{
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        dispGrad.set(0.0);
        // Get the node indices that surround the cell
        IntVector ni[8];
        Vector d_S[8];

        patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
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


        deformed_fiber_vector =pfiberdir[idx]; // not actually deformed yet

        //_______________________UNCOUPLE DEVIATORIC AND DILATIONAL PARTS of DEF GRAD
        //_______________________Ftilde=J^(-1/3)*F and Fvol=J^1/3*Identity

        //_______________________right Cauchy Green (C) tilde and invariants
        rightCauchyGreentilde_new = deformationGradient_new[idx].Transpose()
                                * deformationGradient_new[idx]*pow(J,-(2./3.));
        I1tilde = rightCauchyGreentilde_new.Trace();
        I2tilde = .5*(I1tilde*I1tilde -
                (rightCauchyGreentilde_new*rightCauchyGreentilde_new).Trace());
        I4tilde = Dot(deformed_fiber_vector,
                   (rightCauchyGreentilde_new*deformed_fiber_vector));
        lambda_tilde = sqrt(I4tilde);
        deformed_fiber_vector = deformationGradient_new[idx]*deformed_fiber_vector
                                *(1./lambda_tilde*pow(J,-(1./3.)));
        Matrix3 DY(deformed_fiber_vector,deformed_fiber_vector);

        //________________________________left Cauchy Green (B) tilde
        leftCauchyGreentilde_new = deformationGradient_new[idx]
                     * deformationGradient_new[idx].Transpose()*pow(J,-(2./3.));
        //________________________________hydrostatic pressure term
        p = Bulk*log(J)/J;
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
                           /(lambda_tilde*lambda_tilde);
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

        //_______________________________ assemble Cauchy stress

        pstress[idx] = Identity*p
                     + (leftCauchyGreentilde_new*(c1+c2*I1tilde)
                     - leftCauchyGreentilde_new*leftCauchyGreentilde_new*c2
                     + DY*dWdI4tilde*I4tilde
                     - Identity*(1./3.)
                      *(c1*I1tilde+2.*c2*I2tilde+dWdI4tilde*I4tilde))*2./J;

        //________________________________STIFFNESS = vol. term + Mooney-Rivlin term + Fiber contribution term
        
        double K = Bulk;
        double cc1 = d2WdI4tilde2*I4tilde*I4tilde;
        double cc2MR = (4./3.)*(1./J)*(c1*I1tilde+2.*c2*I2tilde);
        double cc2FC = (4./3.)*(1./J)*dWdI4tilde*I4tilde;

        Matrix3 RB = leftCauchyGreentilde_new;
        Matrix3 RB2 = leftCauchyGreentilde_new*leftCauchyGreentilde_new;
        Matrix3 I = Identity;
        Matrix3 devsMR = (RB*(c1+c2*I1tilde)+RB2*(-c2)+I*(c1*I1tilde+2*c2*I2tilde))*(2./J)*(-2./3.);
        Matrix3 devsFC = (DY-I*(1./3.))*dWdI4tilde*I4tilde*(2./J)*(-2./3.);
        Matrix3 termMR = RB*(1./J)*c2*I1tilde-RB2*(1./J)*c2;

        //________________________________________________________vol. term 

        double cvol[6][6];

        cvol[0][0] = K*(1./J)-2*p;
        cvol[0][1] = K*(1./J);
        cvol[0][2] = K*(1./J);
        cvol[1][1] = K*(1./J)-2*p;
        cvol[1][2] = K*(1./J);
        cvol[2][2] = K*(1./J)-2*p;
        cvol[0][3] =  0.;
        cvol[0][4] =  0.;
        cvol[0][5] =  0.;
        cvol[1][3] =  0.;
        cvol[1][4] =  0.;
        cvol[1][5] =  0.;
        cvol[2][3] =  0.;
        cvol[2][4] =  0.;
        cvol[2][5] =  0.;
        cvol[3][3] = -p;
        cvol[3][4] =  0.;
        cvol[3][5] =  0.;
        cvol[4][4] = -p;
        cvol[4][5] =  0.;
        cvol[5][5] = -p;

        //________________________________________________________Mooney-Rivlin term

        double cMR[6][6];

        cMR[0][0] = (4./J)*c2*RB(0,0)*RB(0,0)-(4./J)*c2*(RB(0,0)*RB(0,0)+RB(0,0)*RB(0,0))
                        +(2./3.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde+devsMR(0,0)+devsMR(0,0)
                        +(-4./3.)*(termMR(0,0)+termMR(0,0));
        cMR[0][1] = (4./J)*c2*RB(0,0)*RB(1,1)-(4./J)*c2*(RB(0,1)*RB(0,1)+RB(0,1)*RB(0,1))
                        +(-1./3.)*cc2MR+devsMR(0,0)+devsMR(1,1)
                        +(-4./3.)*(termMR(0,0)+termMR(1,1));
        cMR[0][2] = (4./J)*c2*RB(0,0)*RB(2,2)-(4./J)*c2*(RB(0,2)*RB(0,2)+RB(0,2)*RB(0,2))
                        +(-1./3.)*cc2MR+devsMR(0,0)+devsMR(2,2)
                        +(-4./3.)*(termMR(0,0)+termMR(2,2));
        cMR[1][1] = (4./J)*c2*RB(1,1)*RB(1,1)-(4./J)*c2*(RB(1,1)*RB(1,1)+RB(1,1)*RB(1,1))
                        +(2./3.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde+devsMR(1,1)+devsMR(1,1)
                        +(-4./3.)*(termMR(1,1)+termMR(1,1));
        cMR[1][2] = (4./J)*c2*RB(1,1)*RB(2,2)-(4./J)*c2*(RB(1,2)*RB(1,2)+RB(1,2)*RB(1,2))
                        +(-1./3.)*cc2MR+devsMR(1,1)+devsMR(2,2)
                        +(-4./3.)*(termMR(1,1)+termMR(2,2));
        cMR[2][2] = (4./J)*c2*RB(2,2)*RB(2,2)-(4./J)*c2*(RB(2,2)*RB(2,2)+RB(2,2)*RB(2,2))
                        +(2./3.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde+devsMR(2,2)+devsMR(2,2)
                        +(-4./3.)*(termMR(2,2)+termMR(2,2));
        cMR[0][3] = (4./J)*c2*RB(0,0)*RB(0,1)-(4./J)*c2*(RB(0,0)*RB(0,1)+RB(0,1)*RB(0,0))
                        +devsMR(0,1)+(-4./3.)*termMR(0,1);
        cMR[0][4] = (4./J)*c2*RB(0,0)*RB(1,2)-(4./J)*c2*(RB(0,1)*RB(0,2)+RB(0,2)*RB(0,1))
                        +devsMR(1,2)+(-4./3.)*termMR(1,2);
        cMR[0][5] = (4./J)*c2*RB(0,0)*RB(2,0)-(4./J)*c2*(RB(0,2)*RB(0,0)+RB(0,0)*RB(0,2))
                        +devsMR(2,0)+(-4./3.)*termMR(2,0);
        cMR[1][3] = (4./J)*c2*RB(1,1)*RB(0,1)-(4./J)*c2*(RB(1,0)*RB(1,1)+RB(1,1)*RB(1,0))
                        +devsMR(0,1)+(-4./3.)*termMR(0,1);
        cMR[1][4] = (4./J)*c2*RB(1,1)*RB(1,2)-(4./J)*c2*(RB(1,1)*RB(1,2)+RB(1,2)*RB(1,1))
                        +devsMR(1,2)+(-4./3.)*termMR(1,2);
        cMR[1][5] = (4./J)*c2*RB(1,1)*RB(2,0)-(4./J)*c2*(RB(1,2)*RB(1,0)+RB(1,0)*RB(1,2))
                        +devsMR(2,0)+(-4./3.)*termMR(2,0);
        cMR[2][3] = (4./J)*c2*RB(2,2)*RB(0,1)-(4./J)*c2*(RB(2,0)*RB(2,1)+RB(2,1)*RB(2,0))
                        +devsMR(0,1)+(-4./3.)*termMR(0,1);
        cMR[2][4] = (4./J)*c2*RB(2,2)*RB(1,2)-(4./J)*c2*(RB(2,1)*RB(2,2)+RB(2,2)*RB(2,1))
                        +devsMR(1,2)+(-4./3.)*termMR(1,2);
        cMR[2][5] = (4./J)*c2*RB(2,2)*RB(2,0)-(4./J)*c2*(RB(2,2)*RB(2,0)+RB(2,0)*RB(2,2))
                        +devsMR(2,0)+(-4./3.)*termMR(2,0);
        cMR[3][3] = (4./J)*c2*RB(0,1)*RB(0,1)-(4./J)*c2*(RB(0,0)*RB(1,1)+RB(0,1)*RB(1,0))
                        +(1./2.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde;
        cMR[3][4] = (4./J)*c2*RB(0,1)*RB(1,2)-(4./J)*c2*(RB(0,1)*RB(1,2)+RB(0,2)*RB(1,1));
        
        cMR[3][5] = (4./J)*c2*RB(0,1)*RB(2,0)-(4./J)*c2*(RB(0,2)*RB(1,0)+RB(0,0)*RB(1,2));

        cMR[4][4] = (4./J)*c2*RB(1,2)*RB(1,2)-(4./J)*c2*(RB(1,1)*RB(2,2)+RB(1,2)*RB(2,1))
                        +(1./2.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde;
        cMR[4][5] = (4./J)*c2*RB(1,2)*RB(2,0)-(4./J)*c2*(RB(1,2)*RB(2,0)+RB(1,0)*RB(2,2));

        cMR[5][5] = (4./J)*c2*RB(2,0)*RB(2,0)-(4./J)*c2*(RB(2,2)*RB(0,0)+RB(2,0)*RB(0,2))
                        +(1./2.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde;

        //________________________________________________________fiber contribution term

        double cFC[6][6];

        cFC[0][0] = (2./3.)*cc2FC+(4./9.)*(1./J)*cc1+devsFC(0,0)+devsFC(0,0)
                        +(-4./3.)*(1./J)*cc1*(DY(0,0)+DY(0,0))+(4./J)*cc1*DY(0,0)*DY(0,0);
        cFC[0][1] = (-1./3.)*cc2FC+devsFC(0,0)+devsFC(1,1)
                        +(-4./3.)*(1./J)*cc1*(DY(0,0)+DY(1,1))+(4./J)*cc1*DY(0,0)*DY(1,1);
        cFC[0][2] = (-1./3.)*cc2FC+devsFC(0,0)+devsFC(2,2)
                        +(-4./3.)*(1./J)*cc1*(DY(0,0)+DY(2,2))+(4./J)*cc1*DY(0,0)*DY(2,2);
        cFC[1][1] = (2./3.)*cc2FC+(4./9.)*(1./J)*cc1+devsFC(1,1)+devsFC(1,1)
                        +(-4./3.)*(1./J)*cc1*(DY(1,1)+DY(1,1))+(4./J)*cc1*DY(1,1)*DY(1,1);
        cFC[1][2] = (-1./3.)*cc2FC+devsFC(1,1)+devsFC(2,2)
                        +(-4./3.)*(1./J)*cc1*(DY(1,1)+DY(2,2))+(4./J)*cc1*DY(1,1)*DY(2,2);
        cFC[2][2] = (2./3.)*cc2FC+(4./9.)*(1./J)*cc1+devsFC(2,2)+devsFC(2,2)
                        +(-4./3.)*(1./J)*cc1*(DY(2,2)+DY(2,2))+(4./J)*cc1*DY(2,2)*DY(2,2);
        cFC[0][3] = devsFC(0,1)+(-4./3.)*(1./J)*cc1*DY(0,1)+(4./J)*cc1*DY(0,0)*DY(0,1);
        cFC[0][4] = devsFC(1,2)+(-4./3.)*(1./J)*cc1*DY(1,2)+(4./J)*cc1*DY(0,0)*DY(1,2);
        cFC[0][5] = devsFC(2,0)+(-4./3.)*(1./J)*cc1*DY(2,0)+(4./J)*cc1*DY(0,0)*DY(2,0);
        cFC[1][3] = devsFC(0,1)+(-4./3.)*(1./J)*cc1*DY(0,1)+(4./J)*cc1*DY(1,1)*DY(0,1);
        cFC[1][4] = devsFC(1,2)+(-4./3.)*(1./J)*cc1*DY(1,2)+(4./J)*cc1*DY(1,1)*DY(1,2);
        cFC[1][5] = devsFC(2,0)+(-4./3.)*(1./J)*cc1*DY(2,0)+(4./J)*cc1*DY(1,1)*DY(2,0);
        cFC[2][3] = devsFC(0,1)+(-4./3.)*(1./J)*cc1*DY(0,1)+(4./J)*cc1*DY(2,2)*DY(0,1);
        cFC[2][4] = devsFC(1,2)+(-4./3.)*(1./J)*cc1*DY(1,2)+(4./J)*cc1*DY(2,2)*DY(1,2);
        cFC[2][5] = devsFC(2,0)+(-4./3.)*(1./J)*cc1*DY(2,0)+(4./J)*cc1*DY(2,2)*DY(2,0);
        cFC[3][3] = (1./2.)*cc2FC+(4./9.)*(1./J)*cc1+(4./J)*cc1*DY(0,1)*DY(0,1);
        cFC[3][4] = (4./J)*cc1*DY(0,1)*DY(1,2);
        cFC[3][5] = (4./J)*cc1*DY(0,1)*DY(2,0);
        cFC[4][4] = (1./2.)*cc2FC+(4./9.)*(1./J)*cc1+(4./J)*cc1*DY(1,2)*DY(1,2);
        cFC[4][5] = (4./J)*cc1*DY(1,2)*DY(2,0);
        cFC[5][5] = (1./2.)*cc2FC+(4./9.)*(1./J)*cc1+(4./J)*cc1*DY(2,0)*DY(2,0);

        //________________________________________________________the STIFFNESS
        
        double D[6][6];

        D[0][0] =cvol[0][0]+cMR[0][0]+cFC[0][0];
        D[0][1] =cvol[0][1]+cMR[0][1]+cFC[0][1];
        D[0][2] =cvol[0][2]+cMR[0][2]+cFC[0][2];
        D[1][1] =cvol[1][1]+cMR[1][1]+cFC[1][1];
        D[1][2] =cvol[1][2]+cMR[1][2]+cFC[1][2];
        D[2][2] =cvol[2][2]+cMR[2][2]+cFC[2][2];
        D[0][3] =cvol[0][3]+cMR[0][3]+cFC[0][3];
        D[0][4] =cvol[0][4]+cMR[0][4]+cFC[0][4];
        D[0][5] =cvol[0][5]+cMR[0][5]+cFC[0][5];
        D[1][3] =cvol[1][3]+cMR[1][3]+cFC[1][3];
        D[1][4] =cvol[1][4]+cMR[1][4]+cFC[1][4];
        D[1][5] =cvol[1][5]+cMR[1][5]+cFC[1][5];
        D[2][3] =cvol[2][3]+cMR[2][3]+cFC[2][3];
        D[2][4] =cvol[2][4]+cMR[2][4]+cFC[2][4];
        D[2][5] =cvol[2][5]+cMR[2][5]+cFC[2][5];
        D[3][3] =cvol[3][3]+cMR[3][3]+cFC[3][3];
        D[3][4] =cvol[3][4]+cMR[3][4]+cFC[3][4];
        D[3][5] =cvol[3][5]+cMR[3][5]+cFC[3][5];
        D[4][4] =cvol[4][4]+cMR[4][4]+cFC[4][4];
        D[4][5] =cvol[4][5]+cMR[4][5]+cFC[4][5];
        D[5][5] =cvol[5][5]+cMR[5][5]+cFC[5][5];


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
      }  // end of loop over particles
    }
  }
  solver->flushMatrix();
}

void
TransIsoHyperImplicit::computeStressTensor(const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
   for(int pp=0;pp<patches->size();pp++){
     const Patch* patch = patches->get(pp);

//
     double p;
     Matrix3 rightCauchyGreentilde_new, leftCauchyGreentilde_new;
     double I1tilde,I2tilde,I4tilde,lambda_tilde;
     double dWdI4tilde, d2WdI4tilde2;
     double shear;
     Vector deformed_fiber_vector;
     Matrix3 deformationGradientInc,dispGrad;

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
     constParticleVariable<double> pvolumeold;
     constParticleVariable<double> ptemperature;
     ParticleVariable<double> pvolume_deformed;
//
     ParticleVariable<double> stretch;
     constParticleVariable<Vector> pvelocity;
     constParticleVariable<Vector> pfiberdir;
     ParticleVariable<Vector> pfiberdir_carry;
//
     constNCVariable<Vector> dispNew;
     delt_vartype delT;

     old_dw->get(delT,lb->delTLabel);
     old_dw->get(px,                  lb->pXLabel,                  pset);
     old_dw->get(pvolumeold,          lb->pVolumeOldLabel,          pset);
     old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);
     old_dw->get(pfiberdir           ,lb->pFiberDirLabel,           pset);//fiber dir the initial one gets passed on

     new_dw->get(dispNew,lb->dispNewLabel,dwi,patch,Ghost::AroundCells,1);//displacement
     new_dw->allocateAndPut(pstress,         lb->pStressLabel_preReloc,   pset);//stress
     new_dw->allocateAndPut(pvolume_deformed,lb->pVolumeDeformedLabel,    pset);//def volume

     //deformation gradient label
     old_dw->get(deformationGradient,        lb->pDeformationMeasureLabel,pset);
     new_dw->allocateAndPut(deformationGradient_new,lb->pDeformationMeasureLabel_preReloc, pset);

     // fiber dir and stretch are only for LV appl.
     new_dw->allocateAndPut(pfiberdir_carry  ,lb->pFiberDirLabel_preReloc,pset);
     new_dw->allocateAndPut(stretch          ,pStretchLabel_preReloc,     pset);

     //_____________________________________________material parameters
     double Bulk  = d_initialData.Bulk;
     //Vector a0 = d_initialData.a0;
     double c1 = d_initialData.c1;
     double c2 = d_initialData.c2;
     double c3 = d_initialData.c3;
     double c4 = d_initialData.c4;
     double c5 = d_initialData.c5;
     double lambda_star = d_initialData.lambda_star;
     double c6 = c3*(exp(c4*(lambda_star-1.))-1.)-c5*lambda_star;

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
     for(ParticleSubset::iterator iter = pset->begin();
                                  iter != pset->end(); iter++){
        particleIndex idx = *iter;

        dispGrad.set(0.0);
        // Get the node indices that surround the cell
        IntVector ni[8];
        Vector d_S[8];

#if 0
        if(ptemperature[idx] < 300.0){
          shear = d_initialData.Shear*2.0;
          bulk  = d_initialData.Bulk *2.0;
        }
        else{
           shear = d_initialData.Shear;
           bulk  = d_initialData.Bulk ;
        }
#endif
        patch->findCellAndShapeDerivatives(px[idx], ni, d_S);

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

        // For diagnostics only
        double I4 = I4tilde*pow(J,(2./3.));
        stretch[idx] = sqrt(I4);
        deformed_fiber_vector = deformationGradient_new[idx]*deformed_fiber_vector*(1./lambda_tilde*pow(J,-(1./3.)));

        Matrix3 DY(deformed_fiber_vector,deformed_fiber_vector);

        //________________________________left Cauchy Green (B) tilde
        leftCauchyGreentilde_new = deformationGradient_new[idx]
                     * deformationGradient_new[idx].Transpose()*pow(J,-(2./3.));

        //________________________________hydrostatic pressure term
        p = Bulk*log(J)/J;

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

        //_______________________________ assemble Cauchy stress
        pstress[idx] = Identity*p
                     + (leftCauchyGreentilde_new*(c1+c2*I1tilde)
                     - leftCauchyGreentilde_new*leftCauchyGreentilde_new*c2
                     + DY*dWdI4tilde*I4tilde
                     - Identity*(1./3.)
                      *(c1*I1tilde+2.*c2*I2tilde+dWdI4tilde*I4tilde))*2./J;
        //________________________________end stress

        pvolume_deformed[idx] = pvolumeold[idx]*J;
      }
     }
   }
}

void TransIsoHyperImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet* patches,
                                                 const bool recursion) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // task->requires(Task::OldDW, lb->pXLabel,      matlset, Ghost::None);
  // new version uses ParentOldDW

  task->requires(Task::ParentOldDW, lb->pXLabel,         matlset,Ghost::None);
  task->requires(Task::ParentOldDW, lb->pVolumeOldLabel, matlset,Ghost::None);
  task->requires(Task::ParentOldDW, lb->pDeformationMeasureLabel,
                                                         matlset,Ghost::None);

  task->requires(Task::ParentOldDW,lb->pTemperatureLabel,matlset,Ghost::None);
  task->requires(Task::OldDW,lb->dispNewLabel,matlset,Ghost::AroundCells,1);
  //
  task->requires(Task::ParentOldDW, lb->pFiberDirLabel, matlset, Ghost::None);
  //
  task->computes(lb->pStressLabel_preReloc,matlset);  
  task->computes(lb->pVolumeDeformedLabel, matlset);
  //
}

void TransIsoHyperImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->pXLabel,                 matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeOldLabel,         matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);

  task->requires(Task::NewDW, lb->dispNewLabel,matlset,Ghost::AroundCells,1);
  task->requires(Task::OldDW, lb->delTLabel);
  //
  task->requires(Task::OldDW, lb->pFiberDirLabel, matlset, Ghost::None);
  //

  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);
  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pFiberDirLabel_preReloc,           matlset);
  task->computes(pStretchLabel_preReloc,                matlset);
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double TransIsoHyperImplicit::computeRhoMicroCM(double pressure, 
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

void TransIsoHyperImplicit::computePressEOSCM(const double rho_cur,double& pressure, 
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

double TransIsoHyperImplicit::getCompressibility()
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
    ASSERTEQ(sizeof(TransIsoHyperImplicit::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(TransIsoHyperImplicit::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "TransIsoHyperImplicit::StateData", true,
                                  &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
