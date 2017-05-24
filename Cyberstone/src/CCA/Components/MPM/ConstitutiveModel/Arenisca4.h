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

#ifndef __Arenisca4_H__
#define __Arenisca4_H__


#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>

#include <cmath>

namespace Uintah {
  class MPMLabel;
  class MPMFlags;

  /**************************************

  ****************************************/

  class Arenisca4 : public ConstitutiveModel {
    // Create datatype for storing model parameters
  public:

    // For usage instructions, see the 'WeibullParser' function
    // header in Kayenta.cc
    struct WeibParameters {
      bool Perturb;           // 'True' for perturbed parameter
      double WeibMed;         // Medain distrib. value OR const value depending on bool Perturb
      int    WeibSeed;        // seed for random number generator
      double WeibMod;         // Weibull modulus
      double WeibRefVol;      // Reference Volume
      std::string WeibDist;   // String for Distribution
    };

    struct CMData {
      double PEAKI1;
      double FSLOPE;
      double STREN;
      double YSLOPE;
      double BETA_nonassociativity;
      double B0;
      double B1;
      double B2;
      double B3;
      double B4;
      double G0;
      double G1;
      double G2;
      double G3;
      double G4;
      double p0_crush_curve;
      double p1_crush_curve;
      double p2_crush_curve;
      double p3_crush_curve;
      double CR;
      double fluid_B0;
      double fluid_pressure_initial;
      double T1_rate_dependence;
      double T2_rate_dependence;
      double subcycling_characteristic_number;
      bool Use_Disaggregation_Algorithm;
	  int J3_type;
	  double J3_psi;
	  double principal_stress_cutoff;
	  
    };
    const VarLabel* pAreniscaFlagLabel;          //0: ok, 1: pevp<-p3
    const VarLabel* pAreniscaFlagLabel_preReloc;
    const VarLabel* pScratchDouble1Label;
    const VarLabel* pScratchDouble1Label_preReloc;
    const VarLabel* pScratchDouble2Label;
    const VarLabel* pScratchDouble2Label_preReloc;
    const VarLabel* pPorePressureLabel;
    const VarLabel* pPorePressureLabel_preReloc;
    const VarLabel* pepLabel;               //Plastic Strain
    const VarLabel* pepLabel_preReloc;
    const VarLabel* pevpLabel;              //Plastic Volumetric Strain
    const VarLabel* pevpLabel_preReloc;
    const VarLabel* peveLabel;              //Elastic Volumetric Strain
    const VarLabel* peveLabel_preReloc;
    const VarLabel* pCapXLabel;
    const VarLabel* pCapXLabel_preReloc;
    const VarLabel* pZetaLabel;
    const VarLabel* pZetaLabel_preReloc;
    const VarLabel* pP3Label;
    const VarLabel* pP3Label_preReloc;
    const VarLabel* pStressQSLabel;
    const VarLabel* pStressQSLabel_preReloc;
    const VarLabel* pScratchMatrixLabel;
    const VarLabel* pScratchMatrixLabel_preReloc;

    // weibull parameter set
    WeibParameters wdist;
    const VarLabel* peakI1IDistLabel;
    const VarLabel* peakI1IDistLabel_preReloc;

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

    CMData d_cm;

    // Prevent copying of this class
    // copy constructor

    Arenisca4& operator=(const Arenisca4 &cm);

    void initializeLocalMPMLabels();

  public:
    // constructor
    Arenisca4(ProblemSpecP& ps, MPMFlags* flag);

    // destructor
    virtual ~Arenisca4();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    Arenisca4* clone();

    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

  private: //non-uintah mpm constitutive specific functions
    int computeStep(const Matrix3& D,
                    const double & Dt,
                    const Matrix3& sigma_n,
                    const double & X_n,
                    const double & Zeta_n,
                    const double & coher,
                    const double & P3,
                    const Matrix3& ep_n,
                    Matrix3& sigma_p,
                    double & X_p,
                    double & Zeta_p,
                    Matrix3& ep_p,
                    long64 ParticleID);

    void computeElasticProperties(double & bulk,
                                  double & shear);

    void computeElasticProperties(const Matrix3 stress,
                                  const Matrix3 ep,
				                  const double& P3,
                                  double & bulk,
                                  double & shear
                                 );

    Matrix3 computeTrialStress(const Matrix3& sigma_old,  // old stress
                               const Matrix3& d_e,        // Strain increment
                               const double& bulk,        // bulk modulus
                               const double& shear);      // shear modulus

    int computeStepDivisions(const double& X,
                             const double& Zeta,
                             const double& P3,
                             const Matrix3& ep,
                             const Matrix3& sigma_n,
                             const Matrix3& sigma_trial);

    void computeInvariants(const Matrix3& stress,
                           Matrix3& S,
                           double& I1,
                           double& J2,
                           double& rJ2,
						   double& J3);

    int computeSubstep(const Matrix3& d_e,       // total strain increment for substep
                       const Matrix3& sigma_old, // stress at start of substep
                       const Matrix3& ep_old,    // plastic strain at start of substep
                       const double & X_old,     // hydrostatic comrpessive strength at start of substep
                       const double & Zeta_old,  // trace of isotropic backstress at start of substep
                       const double & coher,     // scalar valued coher
                       const double & P3,      // initial disaggregation strain
                       Matrix3& sigma_new,    // stress at end of substep
                       Matrix3& ep_new,       // plastic strain at end of substep
                       double & X_new,        // hydrostatic comrpessive strength at end of substep
                       double & Zeta_new      // trace of isotropic backstress at end of substep
                      );

    double computeX(const double& evp, const double& P3);

    double computedZetadevp(double Zeta,
                            double evp);

    double computePorePressure(const double ev);

    int nonHardeningReturn(const Matrix3 & sigma_trial,
                           const Matrix3 & sigma_old,
                           const Matrix3& d_e,
                           const double & X,
                           const double & Zeta,
                           const double & coher, // XXX
                           const double & bulk,
                           const double & shear,
                                 Matrix3 & sigma_new,
                                 Matrix3& d_ep_new);

    void transformedBisection(Vector& sigma_0,
                              const Vector& sigma_trial,
                              const double& X,
                              const double& Zeta,
                              const double& coher,
                              const double  limitParameters[4], // XXX
                              const double& S_star_to_S
                             );

    int transformedYieldFunction(const Matrix3& sigma_star,
                                 const double& X,
                                 const double& Zeta,
                                 const double& coher,
                                 const double  limitParameters[4], // XXX
                                 const double& S_star_to_S
                                );
    int computeYieldFunction(const Matrix3& sigma,
                             const double& X,
                             const double& Zeta,
                             const double& coher,
                             const double  limitParameters[4] // XXX
                            );

    void computeLimitParameters(double *limitParameters,
                                const double& coher //XXX
                               );

    void checkInputParameters();
	
	void computeRotationToSphericalCS(const Vector& pnew,// interior point
		                              const Vector& p0,	 // origin (i.e. trial stress)
		                                    Matrix3& R	 // Rotation matrix
									  );

	void computeEigenProjectors(const Matrix3& A,// Input tensor
								Vector& lambda,  // Ordered eigenvalues {LMH}
								Matrix3& PL,	 // Low eigenprojector
								Matrix3& PM,	 // Mid eigenprojector
								Matrix3& PH		 // High eigenprojector
							   );
	void computeEigenValues(const Matrix3& A,// Input tensor
							Vector& lambda  // Ordered eigenvalues {LMH}
							);
	
	
  public: //Uintah MPM constitutive model specific functions

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
                                               const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion) const;

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

    // Weibull input parser that accepts a structure of input
    // parameters defined as:
    //
    // bool Perturb        'True' for perturbed parameter
    // double WeibMed       Medain distrib. value OR const value
    //                         depending on bool Perturb
    // double WeibMod       Weibull modulus
    // double WeibScale     Scale parameter
    // std::string WeibDist  String for Distribution
    virtual void WeibullParser(WeibParameters &iP);

  };
} // End namespace Uintah


#endif  // __Arenisca4_H__
