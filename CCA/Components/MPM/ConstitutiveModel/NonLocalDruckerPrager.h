/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __NONLOCAL_DRUCKER_PRAGER_H__
#define __NONLOCAL_DRUCKER_PRAGER_H__


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

  class NonLocalDruckerPrager : public ConstitutiveModel {
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
      double alpha;
      double alpha_p;
      double bulk_modulus;
      double k_o;
      double shear_modulus;
      double l_nonlocal;
      double h_local;
      double h_nonlocal;
      double minimum_yield_stress;
      double initial_xstress;
      double initial_ystress;
      double initial_zstress;
      int hardening_type;
    };
    const VarLabel* etaLabel;
    const VarLabel* etaLabel_preReloc;
    const VarLabel* eta_nlLabel;
    const VarLabel* eta_nlLabel_preReloc;
    const VarLabel* pPlasticStrainLabel;  
    const VarLabel* pPlasticStrainLabel_preReloc;  
    const VarLabel* k_o_distLabel;
    const VarLabel* k_o_distLabel_preReloc;
    WeibParameters wdist;
  private:
    CMData d_initialData;
         
    // Prevent copying of this class
    // copy constructor

    NonLocalDruckerPrager& operator=(const NonLocalDruckerPrager &cm);

    void initializeLocalMPMLabels();

  public:
    // constructor
    NonLocalDruckerPrager(ProblemSpecP& ps, MPMFlags* flag);
    NonLocalDruckerPrager(const NonLocalDruckerPrager* cm);
         
    // destructor 
    virtual ~NonLocalDruckerPrager();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone

    NonLocalDruckerPrager* clone();
         
    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);
         
    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    void computeInvariants(Matrix3& stress, Matrix3& S,  double& I1, double& J2);

    void computeInvariants(const Matrix3& stress, Matrix3& S,  double& I1, double& J2);

    double YieldFunction(Matrix3& stress, const double& alpha, double& k_o, double& eta, double& eta_nl, const int& hardening_type);

    double YieldFunction(const Matrix3& stress, const double& alpha, double& k_o, const double& eta, const double& eta_nl, const int& hardening_type);


    double YieldFunction(Matrix3& stress, const double& alpha, double&k_o,const double& eta,const double& eta_nl, const int& hardening_type);

    double alpha_nl(const Point& x, Point& s,const vector<double>& l_nl);

    void EvaluateNonLocalAverage(double& dlambda_nl,double& V_alpha, ParticleVariable<double>& pdlambda, constParticleVariable<Point>& px, NCVariable<double>& gdlambda,NCVariable<double>& gmat,const Patch*& patch, particleIndex& idx, const double& l_nonlocal);




    // carry forward CM data for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);


    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);

    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;
    
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                                   map<const VarLabel*, ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);
    
         
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


#endif  // __NONLOCAL_DRUCKER_PRAGER_H__



