
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <math.h>

using namespace Uintah;

#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

ConstitutiveModel::ConstitutiveModel()
{
}

ConstitutiveModel::~ConstitutiveModel()
{
}

void ConstitutiveModel::computeStressTensor(const PatchSubset*,
					    const MPMMaterial*,
					    DataWarehouse*,
					    DataWarehouse*)

{
}


void ConstitutiveModel::computeStressTensor(const PatchSubset*,
					    const MPMMaterial*,
					    DataWarehouse*,
					    DataWarehouse*,
					    Solver*,
					    const bool)
{
}

void ConstitutiveModel::addComputesAndRequires(Task*, 
					       const MPMMaterial*,
					       const PatchSet*) const
{
}

void ConstitutiveModel::addComputesAndRequires(Task*, 
					       const MPMMaterial*,
					       const PatchSet*,
					       const bool) const
{
}

/////////
// Add initial computes with erosion
void 
ConstitutiveModel::addInitialComputesAndRequiresWithErosion(Task* task,
				     const MPMMaterial* matl,
				     const PatchSet* patches,
				     std::string algorithm)
{
  d_erosionAlgorithm = algorithm;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(lb->pErosionLabel, matlset);
  cout << "Erosion Algorithm = " << d_erosionAlgorithm << endl;

  addInitialComputesAndRequires(task, matl, patches);
}

//////////
// Computes and requires with erosion
void 
ConstitutiveModel::addComputesAndRequiresWithErosion(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->pErosionLabel, matlset, Ghost::None);
  task->computes(lb->pErosionLabel_preReloc, matlset);
  addComputesAndRequires(task, matl, patch);
}

//////////
// Stress update with erosion
void 
ConstitutiveModel::computeStressTensorWithErosion(const PatchSubset* patches,
				const MPMMaterial* matl,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw)
{
  cout << "Using dummy compute stress tensor" << endl;
  computeStressTensor(patches, matl, old_dw, new_dw);
}

void ConstitutiveModel::carryForward(const PatchSubset*,
				     const MPMMaterial*,
				     DataWarehouse*,
				     DataWarehouse*)
{
}

void ConstitutiveModel::carryForwardWithErosion(const PatchSubset* patches,
                                                const MPMMaterial* matl,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  carryForward(patches, matl, old_dw, new_dw);
}

//______________________________________________________________________
//______________________________________________________________________
//          HARDWIRE FOR AN IDEAL GAS -Todd 
double ConstitutiveModel::computeRhoMicro(double press, double gamma,
					  double cv, double Temp)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}

void ConstitutiveModel::computePressEOS(double rhoM, double gamma,
			       double cv, double Temp,
			       double& press, double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;
}
//______________________________________________________________________

Matrix3
ConstitutiveModel::computeVelocityGradient(const Patch* patch,
					   const double* oodx, 
					   const Point& px, 
					   const Vector& psize, 
					   constNCVariable<Vector>& gVelocity) 
{
  // Initialize
  Matrix3 velGrad(0.0);

  // Get the node indices that surround the cell
  IntVector ni[MAX_BASIS];
  Vector d_S[MAX_BASIS];

  patch->findCellAndShapeDerivatives27(px, ni, d_S, psize);

  //cout << "ni = " << ni << endl;
  for(int k = 0; k < d_8or27; k++) {
    //if(patch->containsNode(ni[k])) {
    const Vector& gvel = gVelocity[ni[k]];
    //cout << "GridVel = " << gvel << endl;
    for (int j = 0; j<3; j++){
      for (int i = 0; i<3; i++) {
	velGrad(i+1,j+1) += gvel[i] * d_S[k][j] * oodx[j];
      }
    }
    //}
  }
  //cout << "VelGrad = " << velGrad << endl;
  return velGrad;
}

Matrix3
ConstitutiveModel::computeVelocityGradient(const Patch* patch,
					   const double* oodx, 
					   const Point& px, 
					   constNCVariable<Vector>& gVelocity) 
{
  // Initialize
  Matrix3 velGrad(0.0);

  // Get the node indices that surround the cell
  IntVector ni[MAX_BASIS];
  Vector d_S[MAX_BASIS];

  patch->findCellAndShapeDerivatives(px, ni, d_S);

  for(int k = 0; k < d_8or27; k++) {
    const Vector& gvel = gVelocity[ni[k]];
    //cout << "GridVel = " << gvel << endl;
    for (int j = 0; j<3; j++){
      for (int i = 0; i<3; i++) {
	velGrad(i+1,j+1) += gvel[i] * d_S[k][j] * oodx[j];
      }
    }
  }
  //cout << "VelGrad = " << velGrad << endl;
  return velGrad;
}

Matrix3
ConstitutiveModel::computeVelocityGradient(const Patch* patch,
                                           const double* oodx,
                                           const Point& px,
                                           const Vector& psize,
                                           const short pgFld[],
                                           constNCVariable<Vector>& gVelocity,
                                           constNCVariable<Vector>& GVelocity)
{
  // Initialize
  Matrix3 velGrad(0.0);

  // Get the node indices that surround the cell
  IntVector ni[MAX_BASIS];
  Vector d_S[MAX_BASIS];

  patch->findCellAndShapeDerivatives27(px, ni, d_S, psize);

  Vector gvel;
  //cout << "ni = " << ni << endl;
  for(int k = 0; k < d_8or27; k++) {
    //if(patch->containsNode(ni[k])) {
    if(pgFld[k]==1) gvel = gVelocity[ni[k]];
    if(pgFld[k]==2) gvel = GVelocity[ni[k]];
    //cout << "GridVel = " << gvel << endl;
    for (int j = 0; j<3; j++){
      for (int i = 0; i<3; i++) {
        velGrad(i+1,j+1) += gvel[i] * d_S[k][j] * oodx[j];
      }
    }
    //}
  }
  //cout << "VelGrad = " << velGrad << endl;
  return velGrad;
}

Matrix3
ConstitutiveModel::computeVelocityGradient(const Patch* patch,
                                           const double* oodx,
                                           const Point& px,
                                           const short pgFld[],
                                           constNCVariable<Vector>& gVelocity,
                                           constNCVariable<Vector>& GVelocity)
{
  // Initialize
  Matrix3 velGrad(0.0);

  // Get the node indices that surround the cell
  IntVector ni[MAX_BASIS];
  Vector d_S[MAX_BASIS];

  patch->findCellAndShapeDerivatives(px, ni, d_S);

  Vector gvel;
  for(int k = 0; k < d_8or27; k++) {
    if(pgFld[k]==1)  gvel = gVelocity[ni[k]];
    if(pgFld[k]==2)  gvel = GVelocity[ni[k]];
    //cout << "GridVel = " << gvel << endl;
    for (int j = 0; j<3; j++){
      for (int i = 0; i<3; i++) {
        velGrad(i+1,j+1) += gvel[i] * d_S[k][j] * oodx[j];
      }
    }
  }
  //cout << "VelGrad = " << velGrad << endl;
  return velGrad;
}

// Convert J-integral into stress intensity factors
void 
ConstitutiveModel::ConvertJToK(const MPMMaterial*,
     const Vector&,const Vector&,const Vector&,Vector& SIF)
{
  SIF=Vector(-9999.,-9999.,-9999.);
}

// Calculate polar decomposition using Simo page 244
void 
ConstitutiveModel::polarDecomposition(const Matrix3& F, 
                                      Matrix3& R,
                                      Matrix3& U) const
{
  Matrix3 C = F.Transpose()*F;
  double I1 = C.Trace();
  Matrix3 Csq = C*C;
  double I2 = .5*(I1*I1 - Csq.Trace());
  double I3 = C.Determinant();
  double b = I2 - (I1*I1)/3.0;
  double c = -(2.0/27.0)*I1*I1*I1 + (I1*I2)/3.0 - I3;
  double TOL3 = 1e-8;
  double x[4];

  if(fabs(b) <= TOL3){
    c = Max(c,0.); x[1] = -pow(c,1./3.); x[2] = x[1]; x[3] = x[1];
  } else {
    //	cout << "c = " << c << endl;
    double m = 2.*sqrt(-b/3.);
    //	cout << "m = " << m << endl;
    double n = (3.*c)/(m*b);
    //	cout << "n = " << n << endl;
    if (fabs(n) > 1.0) n = (n/fabs(n));
    double t = atan(sqrt(1-n*n)/n)/3.0;
    //	cout << "t = " << t << endl;
    for(int i=1;i<=3;i++){
      x[i] = m * cos(t + 2.*(((double) i) - 1.)*M_PI/3.);
      //	  cout << "x[i] = " << x[i] << endl;
    }
  }
  double lam[4];
  for(int i=1;i<=3;i++) lam[i] = sqrt(x[i] + I1/3.0);

  double i1 = lam[1] + lam[2] + lam[3];
  double i2 = lam[1]*lam[2] + lam[1]*lam[3] + lam[2]*lam[3];
  double i3 = lam[1]*lam[2]*lam[3];
  double D = i1*i2 - i3;
  //      cout << "D = " << D << endl;

  Matrix3 One; One.Identity();
  U = (C*(i1*i1-i2) + One*i1*i3 - Csq)*(1./D);
  Matrix3 Uinv = (C - U*i1 + One*i2)*(1./i3);
  R = F*Uinv;
  //      cout << U << endl << endl;
  //      cout << R << endl << endl;
}
