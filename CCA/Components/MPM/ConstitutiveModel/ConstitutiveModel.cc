
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
// Initialize CM data with erosion 
void 
ConstitutiveModel::initializeCMDataWithErosion(const Patch* patch,
			     const MPMMaterial* matl,
			     DataWarehouse* new_dw)
{
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(),
						   patch);
  ParticleVariable<double> pErosion;
  new_dw->allocateAndPut(pErosion, lb->pErosionLabel, pset);
  ParticleSubset::iterator iter = pset->begin();
  for (; iter != pset->end(); iter++) {
    pErosion[*iter] = 1.0;
  }

  initializeCMData(patch, matl, new_dw);
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

