
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

using namespace Uintah;

ConstitutiveModel::ConstitutiveModel()
{
}

ConstitutiveModel::~ConstitutiveModel()
{
}


void ConstitutiveModel::computeStressTensorImplicit(const PatchSubset*,
						    const MPMMaterial*,
						    DataWarehouse*,
						    DataWarehouse*,
						    SparseMatrix<double,int>&,
#ifdef HAVE_PETSC
						    Mat&,
						    map<const Patch*, Array3<int> >& d_petscLocalToGlobal,
#endif
						    const bool)
{
}


void ConstitutiveModel::computeStressTensorImplicitOnly(const PatchSubset*,
							const MPMMaterial*,
							DataWarehouse*,
							DataWarehouse*)
{
}

void ConstitutiveModel::addComputesAndRequiresImplicit(Task*, 
						       const MPMMaterial*,
						       const PatchSet*,
						       const bool)
{

}


void ConstitutiveModel::addComputesAndRequiresImplicitOnly(Task*,
							   const MPMMaterial*,
							   const PatchSet*,
							   const bool) 
{
}

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

//
// Polar decomposition using the deformation gradient
// Returns the Cauchy-Green Strain Tensor (left or right)
// Calculates the stretch and rotation tensors after polar
// decomposition and returns updated values
//
void
ConstitutiveModel::polarDecomposition(const Matrix3& defGrad,
				      Matrix3& cauchyGreen,
                                      Matrix3& stretch,
                                      Matrix3& rotation,
                                      double tolerance,
				      int leftOrRight)
{
  // Calculate the left (C) or right (b) Cauchy-Green tensor 
  // where C = Ftranspose*F and b = F*Ftranspose
  if (leftOrRight == RIGHT_POLAR)  
    cauchyGreen = (defGrad.Transpose())*defGrad;
  else
    cauchyGreen = defGrad*(defGrad.Transpose());

  // Find the principal invariants of the left or right Cauchy-Green tensor (b) 
  // where b = F*Ftransposeb
  double I1 = cauchyGreen.Trace();
  double I1Square = I1*I1;
  Matrix3 cauchyGreenSquare = cauchyGreen*cauchyGreen;
  double I2 = 0.5*(I1Square - cauchyGreenSquare.Trace());
  double I3 = cauchyGreen.Determinant();

  // Find the squares of the principal stretches lambdaA, lambdaB, lambdaC
  // or lambdaSq(ii), ii = 1..3
  double lambdaSq[3];
  double oneThird = 1.0/3.0;
  double oneThirdI1 = oneThird*I1;
  double bb = I2 - oneThird*I1Square;
  double cc = -(2.0/9.0)*I1Square*oneThirdI1 + oneThirdI1*I2 - I3;
  if (fabs(bb) > tolerance) {
    double mm = 2.0*sqrt(-oneThird*bb);
    double nn = 3.0*cc/(mm*bb);
    double tt = oneThird*atan(sqrt(1-(nn*nn))/nn);
    for (int ii = 1; ii < 4; ++ii) 
      lambdaSq[ii] = mm*cos(tt+2.0*(double)(ii-1)*oneThird*M_PI) + oneThirdI1; 
  } else {
    for (int ii = 1; ii < 4; ++ii) 
      lambdaSq[ii] = -pow(cc, oneThird) + oneThirdI1; 
  }

  // Find the stretch tensor 
  Matrix3 one;
  one.Identity();
  double lambda2p3 = lambdaSq[2] + lambdaSq[3];
  double lambda2m3 = lambdaSq[2]*lambdaSq[3];
  double i1 = lambdaSq[1] + lambda2p3;
  double i2 = lambdaSq[1]*lambda2p3 + lambda2m3;
  double i3 = lambdaSq[1]*lambda2m3;
  double DD = i1*i2 - i3;
  stretch.Identity();
  rotation.Identity();
  Matrix3 stretchInv;
  if (fabs(DD) > tolerance && fabs(i3) > tolerance) {
    stretch = (cauchyGreenSquare*(-1) + cauchyGreen*(i1*i1-i2) + one*(i1*i3))*(1.0/DD);
    stretchInv = (cauchyGreen - (stretch*i1 - one*i2))*(1.0/i3);

    // Calculate the rotation tensor
    if (leftOrRight == RIGHT_POLAR)  
      rotation = defGrad*stretchInv;
    else
      rotation = stretchInv*defGrad;
  }
}
