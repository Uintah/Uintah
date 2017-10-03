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

#ifndef __ARENISCA_H__
#define __ARENISCA_H__


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

  class Arenisca : public ConstitutiveModel {
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
      bool Use_Disaggregation_Algorithm;
      double FSLOPE;
      double CR;
      double p0_crush_curve;
      double p1_crush_curve;
      double p3_crush_curve;
      double p4_fluid_effect;
      double fluid_B0;
      double subcycling_characteristic_number;
      double PEAKI1;
      double B0;
      double G0;
      double FSLOPE_p;
      double hardening_modulus;
      double kinematic_hardening_constant;
      double fluid_pressure_initial;
      double gruneisen_parameter;
      double T1_rate_dependence;
      double T2_rate_dependence;
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
    const VarLabel* pevvLabel;              //EG: Disaggregation Volumetric Strain
    const VarLabel* pevvLabel_preReloc;
    const VarLabel* pev0Label;              //JG: Initial Disaggregation Volumetric Strain
    const VarLabel* pev0Label_preReloc;
    const VarLabel* peqpsLabel;              //Plastic Volumetric Strain
    const VarLabel* peqpsLabel_preReloc;
    const VarLabel* peveLabel;              //Elastic Volumetric Strain
    const VarLabel* peveLabel_preReloc;
    const VarLabel* pCapXLabel;
    const VarLabel* pCapXLabel_preReloc;
    const VarLabel* pCapXDYLabel;
    const VarLabel* pCapXDYLabel_preReloc;
    const VarLabel* pKappaLabel;
    const VarLabel* pKappaLabel_preReloc;
    const VarLabel* pZetaLabel;
    const VarLabel* pZetaLabel_preReloc;
    const VarLabel* pZetaDYLabel;
    const VarLabel* pZetaDYLabel_preReloc;
    const VarLabel* pIotaLabel;
    const VarLabel* pIotaLabel_preReloc;
    const VarLabel* pIotaDYLabel;
    const VarLabel* pIotaDYLabel_preReloc;
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
           sqrt_three,
           one_sqrt_three,
           small_number,
           big_number;
    CMData d_cm;

    // Prevent copying of this class
    // copy constructor

    Arenisca& operator=(const Arenisca &cm);

    void initializeLocalMPMLabels();

  public:
    // constructor
    Arenisca(ProblemSpecP& ps, MPMFlags* flag);

    // destructor
    virtual ~Arenisca();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone

    Arenisca* clone();

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
    int computeStressTensorStep(const Matrix3& trial_stress,
                                Matrix3& sigma_new,
                                Matrix3& ep_new,
                                double& evp_new,
                                double& eve_new,
                                double& X_new,
                                double& Kappa_new,
                                double& Zeta_new,
                                double& bulk,
                                double& PEAKI1,
                                long64 ParticleID);

    void computeInvariants(const Matrix3& stress,
                           Matrix3& S,
                           double& I1,
                           double& J2);


    double YieldFunction(const double& I1,
                         const double& J2,
                         const double& X,
                         const double& Zeta,
                         const double& threeKby2G,
                         const double& PEAKI1);

    double ComputeNonHardeningReturn(const double& R,
                                     const double& Z,
                                     const double& CapX,
                                     const double& Beta,
                                     double& r_new,
                                     double& z_new);

    double TransformedYieldFunction(const double& R,
                                    const double& Z,
                                    const double& X,
                                    const double& Beta,
                                    const double& PEAKI1);

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
    
  private:  //Non-Uintah MPM constitutive model class functions
    double computeev0();

    double computedfdKappa(double I1,
                           double X,
                           double Kappa,
                           double Zeta);

    double computedfdZeta(double I1,
                          double X,
                          double Kappa,
                          double Zeta);

    double computeBulkModulus(double ev);

    double computeX(double evp);

    double computedZetadevp(double Zeta,
                            double evp);
  };
} // End namespace Uintah


#endif  // __ARENISCA_H__



