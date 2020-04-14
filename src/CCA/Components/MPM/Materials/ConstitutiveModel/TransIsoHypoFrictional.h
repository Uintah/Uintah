/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

//  TransIsoHypoFrictional.h
//  class ConstitutiveModel ConstitutiveModel data type -- 3D -
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for a Transversely Isotropic Hyperelastic material
//    Features:
//      Usage:



#ifndef __TransIsoHypoFrictional_CONSTITUTIVE_MODEL_H__
#define __TransIsoHypoFrictional_CONSTITUTIVE_MODEL_H__


#include <Core/Disclosure/TypeDescription.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <cmath>
#include <vector>

namespace Uintah {
  class TransIsoHypoFrictional : public ConstitutiveModel {
  private:
    // Create datatype for storing model parameters
    bool d_useModifiedEOS;
  public:
    struct CMData {   //_________________________________________modified here
      double E_t;						// transverse modulus
      double Y_t;						// transverse cohesive strength
      double E_a;					  // axial modulus
      double nu_at;			  	// axial-transverse poisson ratio
      double nu_t;					// transverse poisson ratio
      double G_at;					// axial-transverse shear modulus
      double mu_fiber;			// inter-fiber friction coefficient
      double crimp_stretch; // stretch to uncrimp fibers
      double crimp_ratio;   // ratio of uncrimped to initial tensile modulus.
      double phi_0;         // initial micro porosity in fiber bundle
      double alpha;         // exponent in porosity scaling of transverse moduli: (1-phi)^alpha
      double bulk;          // nominal bulk modulus for MPM-ICE cell centered compressibility calculations
      Vector n_fiber;       // unit vector in fiber direction
    };

    const VarLabel* pStretchLabel;           // Current fiber stretch, For diagnostics
    const VarLabel* pStretchLabel_preReloc;  // For diagnostic

    const VarLabel* pPorosityLabel;           // Current nominal porosity, For diagnostics
    const VarLabel* pPorosityLabel_preReloc;  // For diagnostic

    const VarLabel* pCrimpLabel;              // Current stretch-scaleing factor For diagnostics
    const VarLabel* pCrimpLabel_preReloc;     // For diagnostic

  private:
    double one_third,
           two_third,
           four_third,
           sqrt_two,
           one_sqrt_two,
           sqrt_three,
           one_sqrt_three,
           one_sixth,
           one_ninth,
           pi,
           pi_fourth,
           pi_half,
           small_number,
           big_number,
           Kf,
           Km,
           phi_i,
           ev0,
           C1;

    Matrix3 Identity;
    Matrix3 Zero;

    CMData d_cm;

    // Prevent copying of this class
    // copy constructor
    TransIsoHypoFrictional& operator=(const TransIsoHypoFrictional &cm);

    // Basis function for transverse isotropy B3(n)_ijpq
    double B1(const Vector& n, // fiber direction
		      const int& i,    // index
		      const int& j,    // index
		      const int& p,    // index
		      const int& q);    // index
    // Basis function for transverse isotropy B3(n)_ijpq
    double B2(const Vector& n, // fiber direction
		      const int& i,    // index
		      const int& j,    // index
		      const int& p,    // index
		      const int& q);    // index
    // Basis function for transverse isotropy B3(n)_ijpq
    double B3(const Vector& n, // fiber direction
		      const int& i,    // index
		      const int& j,    // index
		      const int& p,    // index
		      const int& q);    // index
    // Basis function for transverse isotropy B3(n)_ijpq
    double B4(const Vector& n, // fiber direction
		      const int& i,    // index
		      const int& j,    // index
		      const int& p,    // index
		      const int& q);    // index
    // Basis function for transverse isotropy B3(n)_ijpq
    double B5(const Vector& n, // fiber direction
		      const int& i,    // index
		      const int& j,    // index
		      const int& p,    // index
		      const int& q);    // index

    // estimates porosity assuming no matrix compression.
    double computePorosity(const double& J);

    // computes crimp scale factor based on fiber stretch
    double computeCrimp(const double& stretch);

    // compute bulk modulus based on scale Trans Isotropic elastic properties.
    double computeBulkModulus(const double& J,      // volumetric stretch
    		                      const double& stretch // fiber stretch
    );

    // cubic blending function
    double smoothStep(const double& x,
    		const double& xmin,
    		const double& xmax);

  public:
    // constructors
    TransIsoHypoFrictional( ProblemSpecP& ps, MPMFlags* flag );

    // destructor
    virtual ~TransIsoHypoFrictional();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    TransIsoHypoFrictional* clone();

    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);


    // carry forward CM data for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);

    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet*) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion) const;

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

    virtual Vector getInitialFiberDir();

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

  };
} // End namespace Uintah



#endif  // __TransIsoHypoFrictional_CONSTITUTIVE_MODEL_H__

