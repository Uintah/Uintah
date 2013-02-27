/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

//  UCNH.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for Compressible NeoHookean materials
//    Features:
//      Usage:


#ifndef __UNIFIED_NEOHOOK_CONSTITUTIVE_MODEL_H__
#define __UNIFIED_NEOHOOK_CONSTITUTIVE_MODEL_H__

namespace Uintah {
  // Structures for Plasticitity 

struct UCNHStateData {
    double Alpha;
  };
  class TypeDescription;
  const TypeDescription* fun_getTypeDescription(UCNHStateData*);
}

#include <Core/Util/Endian.h>

namespace SCIRun {
  using namespace Uintah;
  inline void swapbytes( Uintah::UCNHStateData& d)
  { swapbytes(d.Alpha); }
} // namespace SCIRun

#include "ConstitutiveModel.h"  
#include "ImplicitCM.h"
#include "PlasticityModels/MPMEquationOfState.h"
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <cmath>
#include <vector>

namespace Uintah {
  // Classes needed by UCNH
  class TypeDescription;
    
  class UCNH : public ConstitutiveModel, public ImplicitCM {

  ///////////////
  // Variables //
  ///////////////
  public:

    // Basic Requirements //
    ////////////////////////
    // Create datatype for storing model parameters
    struct CMData {
      double Bulk;
      double tauDev;
      // For Plasticity
      double FlowStress;
      double K;
      double Alpha;
    };
      
    struct YieldDistribution {
      std::string dist;
      double range;
      int seed;
    };

    // Damage Requirements //
    /////////////////////////
    // Create datatype for failure strains
    struct FailureStressOrStrainData {
      double mean;         /* Mean failure stress, strain or cohesion */
      double std;          /* Standard deviation of failure strain */
                           /* or Weibull modulus */
      double exponent;     /* Exponent used in volume scaling of failure crit */
      double refVol;       /* Reference volume for scaling failure criteria */
      std::string scaling; /* Volume scaling method: "none" or "kayenta" */
      std::string dist;    /* Failure distro: "constant", "gauss" or "weibull"*/
      int seed;            /* seed for random number distribution generator */
      double t_char;       /* characteristic time for damage to occur */
    };

    //Create datatype for brittle damage
    struct BrittleDamageData {
      double r0b;          /* Initial energy threshold (\sqrt{Pa}) */
      double Gf;           /* Fracture energy (J/m^3) */
      double constant_D;   /* Shape factor in softening function */
      double maxDamageInc; /* Maximum damage increment in a time step */
      bool allowRecovery;  /* Recovery of stiffness allowed */
      double recoveryCoeff;  /* Fraction of stiffness to be recovered */
      bool printDamage;    /* Flag to print damage */
    };

    const VarLabel* bElBarLabel;
    const VarLabel* bElBarLabel_preReloc;
    const VarLabel* pFailureStressOrStrainLabel;
    const VarLabel* pLocalizedLabel;
    const VarLabel* pDamageLabel;
    const VarLabel* pDeformRateLabel;
    const VarLabel* pTimeOfLocLabel;
    const VarLabel* pFailureStressOrStrainLabel_preReloc;
    const VarLabel* pLocalizedLabel_preReloc;
    const VarLabel* pDamageLabel_preReloc;
    const VarLabel* pDeformRateLabel_preReloc;
    const VarLabel* pTimeOfLocLabel_preReloc;

    // Plasticity Requirements //
    /////////////////////////////
    const VarLabel* pPlasticStrain_label;
    const VarLabel* pPlasticStrain_label_preReloc;
    const VarLabel* pYieldStress_label;
    const VarLabel* pYieldStress_label_preReloc;
      
  protected:
    // Flags indicating if damage and/or plasticity should be used
    bool d_useDamage;
    bool d_usePlasticity;
      
    // Basic Requirements //
    ////////////////////////
    CMData d_initialData;
    bool d_useModifiedEOS; 
    int d_8or27;
      
    // Damage Requirments //
    ////////////////////////
    YieldDistribution d_yield;
    FailureStressOrStrainData d_epsf;
    BrittleDamageData d_brittle_damage;
      
    // Erosion algorithms
    bool d_setStressToZero; /* set stress tensor to zero*/
    bool d_allowNoTension;  /* retain compressive mean stress after failue*/
    bool d_allowNoShear;    /* retain mean stress after failure - no deviatoric stress */
                            /* i.e., no deviatoric stress */
    bool d_brittleDamage;   /* use brittle damage with mesh size control*/

    std::string d_failure_criteria; /* Options are:  "MaximumPrincipalStrain" */
                                    /* "MaximumPrincipalStress", "MohrColoumb"*/

    // These three are for the MohrColoumb option
    double d_friction_angle;  // Assumed to come in degrees
    double d_tensile_cutoff;  // Fraction of the cohesion at which 
                              // tensile failure occurs

    // Initial stress state
    bool d_useInitialStress;
    double d_init_pressure;  // Initial pressure

    // Model factories
    //bool d_useEOSFactory;
    MPMEquationOfState* d_eos;

  ///////////////
  // Functions //
  ///////////////
  private:
    // Prevent copying of this class
    // copy constructor
    //UCNH(const UCNH &cm);
    UCNH& operator=(const UCNH &cm);
      
    // Plasticity requirements
    //friend const TypeDescription* fun_getTypeDescriptiont(StateData*);

  public:
    // constructors
    UCNH(ProblemSpecP& ps, MPMFlags* flag);
    UCNH(ProblemSpecP& ps, MPMFlags* flag, bool plas, bool dam);
    UCNH(const UCNH* cm);

    // specifcy what to output from the constitutive model to an .xml file
    virtual void outputProblemSpec(ProblemSpecP& ps, bool output_cm_tag = true);
    
    // clone
    UCNH* clone();
      
    // destructor
    virtual ~UCNH();

    // carry forward CM data for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);
    
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);
    
    
    // Scheduling Functions //
    //////////////////////////
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;
    
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion,
                                        const bool schedPar = true) const;
    
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;
    
    ////////////////////////////////////////////////////////////////////////
    /*! \\brief Add the requires for failure simulation. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addRequiresDamageParameter(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches) const;
    
    
    // Compute Functions //
    ///////////////////////
    // main computation of pressure from constitutive model's equation of state
    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const MPMMaterial* matl,
                                   double temperature);
    
    // main computation of density from constitutive model's equation of state
    virtual double computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl,
                                     double temperature,
                                     double rho_guess);
    
    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);
    
    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);
    
    // Damage specific CST for solver
    virtual void computeStressTensorImplicit(const PatchSubset* patches,
                                             const MPMMaterial* matl,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw,
                                             Solver* solver,
                                             const bool );
    
    
    // Helper Functions //
    //////////////////////
    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);
    
    // Returns the compressibility of the material
    virtual double getCompressibility();
      
    ////////////////////////////////////////////////////////////////////////
    /*! \\brief Get the flag that marks a failed particle. */
    ////////////////////////////////////////////////////////////////////////
    virtual void getDamageParameter(const Patch* patch, 
                                    ParticleVariable<int>& damage, int dwi,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw);
    
    
  private:
    // Damage requirements //
    /////////////////////////
    void getYieldStressDistribution(ProblemSpecP& ps);

    void setYieldStressDistribution(const UCNH* cm);

    void getFailureStressOrStrainData(ProblemSpecP& ps);

    void getBrittleDamageData(ProblemSpecP& ps);

    void setFailureStressOrStrainData(const UCNH* cm);      

    void setBrittleDamageData(const UCNH* cm);
      
    void initializeLocalMPMLabels();
      
    void setErosionAlgorithm();
      
    void setErosionAlgorithm(const UCNH* cm);
      
  protected:
    // compute stress at each particle in the patch
    void computeStressTensorImplicit(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);
      
    // Modify the stress if particle has failed
    void updateFailedParticlesAndModifyStress(const Matrix3& FF, 
                                              const double& pFailureStrain, 
                                              const int& pLocalized,
                                              int& pLocalized_new, 
                                              const double& pTimeOfLoc,
                                              double& pTimeOfLoc_new, 
                                              Matrix3& pStress_new,
                                              const long64 particleID,
                                              double time);

    // Modify the stress for brittle damage
    void updateDamageAndModifyStress(const Matrix3& FF, 
                                     const double& pFailureStrain, 
                                     double& pFailureStrain_new, 
                                     const double& pVolume, 
                                     const double& pDamage,
                                     double& pDamage_new, 
                                     Matrix3& pStress_new,
                                     const long64 particleID);
      
    /*! Compute tangent stiffness matrix */
    void computeTangentStiffnessMatrix(const Matrix3& sigDev, 
                                       const double&  mubar,
                                       const double&  J,
                                       const double&  bulk,
                                       double D[6][6]);
    /*! Compute BT*Sig*B (KGeo) */
    void BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
                    double BnTsigBn[24][24]) const;
      
    /*! Compute K matrix */
    void computeStiffnessMatrix(const double B[6][24],
                                const double Bnl[3][24],
                                const double D[6][6],
                                const Matrix3& sig,
                                const double& vol_old,
                                const double& vol_new,
                                double Kmatrix[24][24]);
  };
} // End namespace Uintah

#endif  // __UNIFIED_NEOHOOK_CONSTITUTIVE_MODEL_H__ 
