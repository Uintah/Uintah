/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

// CeramicFiberDamage.h
//
// Cross-ply ceramic-matrix composite with progressive fiber and matrix damage.
// Micromechanics: Tandon-Weng (1984) Eshelby inclusion theory for porous matrix
// elasticity, Hashin-Rosen mixing rules for fiber composite, cross-ply averaging
// by 90-degree rotation about the fiber-plane normal.
//
// Damage state per particle:
//   rho  - matrix crack density (two in-plane directions)
//   nu   - crack-opening fraction (two directions)
//   psi  - fiber fracture factor (1 = intact, 0 = fully fractured)
//
// Stress: 2nd Piola-Kirchhoff from Green-Lagrange strain, pushed forward to
// Cauchy via sigma = (1/J) F S F^T.

#ifndef __CERAMIC_FIBER_DAMAGE_CONSTITUTIVE_MODEL_H__
#define __CERAMIC_FIBER_DAMAGE_CONSTITUTIVE_MODEL_H__

#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <vector>

namespace Uintah {

class CeramicFiberDamage : public ConstitutiveModel {
public:

  struct CMData {
    // Matrix elastic constants
    double Em;   // Young's modulus (Pa)
    double vm;   // Poisson's ratio

    // Fiber elastic constants
    double Eft;  // transverse modulus (Pa)
    double Efl;  // longitudinal modulus (Pa)
    double Gfl;  // longitudinal shear modulus (Pa)
    double Kf;   // bulk modulus (Pa)
    double vf;   // Poisson's ratio

    // Microstructure
    double fiberVF;          // fiber volume fraction (0-1)
    double porosity;         // matrix porosity (0-1)
    double poreAspectRatio;  // pore aspect ratio; negative = spherical

    // Damage model constants (16 parameters from Python material_constants list)
    double alphaM;         // [0]  matrix thermal expansion coefficient (1/K)
    double sigO;           // [1]  reference matrix stress (Pa)
    double rhoO;           // [2]  initial crack density
    double rhoC;           // [3]  crack growth rate coefficient
    double rhoN;           // [4]  crack growth rate exponent
    double nuO;            // [5]  initial crack-opening fraction
    double nuN;            // [6]  crack-opening exponent
    double vFEff;          // [7]  effective fiber VF for damage calc
    double EFiber;         // [8]  fiber modulus used in damage (Pa)
    double EMatrix;        // [9]  matrix modulus used in damage (Pa)
    double residualStress; // [10] residual matrix stress R_m (Pa)
    double TRef;           // [11] reference temperature (K)
    double psiC;           // [12] fiber fracture log-rate coefficient
    double psiN;           // [13] fiber fracture exponent
    double epsO;           // [14] fiber fracture onset GL strain
    double epsF;           // [15] fiber fracture failure GL strain

    // Precomputed undamaged cross-ply engineering constants (set in constructor)
    // Ordering: [E11, E22, E33, G23, G13, G12, v12, v13, v23]
    double moduli[9];

    // Effective bulk modulus derived from undamaged stiffness (for EOS)
    double bulkMod;
  };

private:
  CMData d_initialData;

  // Per-particle damage state: each stored as Vector with x=dir0, y=dir1, z=0
  const VarLabel* pCrackDensityLabel;
  const VarLabel* pCrackDensityLabel_preReloc;
  const VarLabel* pCrackOpeningLabel;
  const VarLabel* pCrackOpeningLabel_preReloc;
  const VarLabel* pFiberFractureLabel;
  const VarLabel* pFiberFractureLabel_preReloc;

  CeramicFiberDamage& operator=(const CeramicFiberDamage&);

public:
  CeramicFiberDamage(ProblemSpecP& ps, MPMFlags* flag);
  virtual ~CeramicFiberDamage();

  virtual void outputProblemSpec(ProblemSpecP& ps, bool output_cm_tag = true);

  CeramicFiberDamage* clone();

  virtual void computeStableTimeStep(const Patch* patch,
                                     const MPMMaterial* matl,
                                     DataWarehouse* new_dw);

  virtual void computeStressTensor(const PatchSubset* patches,
                                   const MPMMaterial* matl,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);

  virtual void carryForward(const PatchSubset* patches,
                            const MPMMaterial* matl,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw);

  virtual void initializeCMData(const Patch* patch,
                                const MPMMaterial* matl,
                                DataWarehouse* new_dw);

  virtual void addComputesAndRequires(Task* task,
                                      const MPMMaterial* matl,
                                      const PatchSet* patches) const;

  virtual void addComputesAndRequires(Task* task,
                                      const MPMMaterial* matl,
                                      const PatchSet* patches,
                                      const bool recursion,
                                      const bool SchedParent = true) const;

  virtual void addParticleState(std::vector<const VarLabel*>& from,
                                std::vector<const VarLabel*>& to);

  virtual double computeRhoMicroCM(double pressure,
                                   const double p_ref,
                                   const MPMMaterial* matl,
                                   double temperature,
                                   double rho_guess);

  virtual void computePressEOSCM(double rho_m, double& press_eos,
                                 double p_ref,
                                 double& dp_drho, double& ss_new,
                                 const MPMMaterial* matl,
                                 double temperature);

  virtual double getCompressibility();
};

} // End namespace Uintah

#endif // __CERAMIC_FIBER_DAMAGE_CONSTITUTIVE_MODEL_H__
