/*

The MIT License

Copyright (c) 2013-2014 The Johns Hopkins University

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


// Adapted from UCNH.cc by Andy Tonge Dec 2011

//  
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for Compressible NeoHookean materials
//    Features:
//      Usage:


#ifndef __TONGE_RAMESH_CONSTITUTIVE_MODEL_H__
#define __TONGE_RAMESH_CONSTITUTIVE_MODEL_H__

namespace Uintah {
  // Structures for Plasticitity 

struct TongeRameshStateData {
    double Alpha;
  };
  class TypeDescription;
  const TypeDescription* fun_getTypeDescription(TongeRameshStateData*);
}

#include <Core/Util/Endian.h>

namespace SCIRun {
  using namespace Uintah;
  inline void swapbytes( Uintah::TongeRameshStateData& d)
  { swapbytes(d.Alpha); }
} // namespace SCIRun
#include "ConstitutiveModel.h"  
#include "ImplicitCM.h"
#include "PlasticityModels/MPMEquationOfState.h" // Including this file causes an error
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <cmath>
#include <vector>
#include <Core/Math/MersenneTwister.h>
// #include <iostream>
// #include <stdexcept>

namespace Uintah {
  // Classes needed by TongeRamesh
  class TypeDescription;
    
  class TongeRamesh : public ConstitutiveModel, public ImplicitCM {

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
      double timeConstant;
    };
      
    const VarLabel* bElBarLabel;
    const VarLabel* bElBarLabel_preReloc;

    // Variables for the new damage model:
    VarLabel** wingLengthLabel_array;
    VarLabel** wingLengthLabel_array_preReloc;
    VarLabel** starterFlawSize_array;
    VarLabel** starterFlawSize_array_preReloc;
    VarLabel** flawNumber_array;
    VarLabel** flawNumber_array_preReloc;

    // Create a datatype for the flaw distribution:
    struct flawDistributionData {
      int numCrackFamilies;     // Number of families to discritize the distribution into
      double meanFlawSize;      // Mean size of the flaws
      double flawDensity;       // Mean flaw density in the sample
      double stdFlawSize;       // Standard deviation of the flaw size
      std::string type;              // Type of the distribution (delta, normal, pareto)
      double minFlawSize;       // Minimum flaw for Pareto dist
      double maxFlawSize;       // Maximum flaw for Pateto dist
      double exponent;          // Exponent for Pateto dist
      bool   randomizeDist;     // Make each particle have a unique distribution
      int    randomSeed;        // Seed for random number generation
      int    randomMethod;      // Method for selecting bin size and location
      double binBias;           // Exponent for flaw distribution bin bias (1.0 for no bias)
      bool   useEtaField;       // Flag for using a fourier field to define the local flaw density
      std::string etaFilename;       // File name containing fourier data for flaw density
      bool   useSizeField;      // Flag for using a fourier field to define the local flaw size shift
      std::string sizeFilename;       // File name containing fourier data for flaw size
      int Ncutoff;		 // Cutoff for Poisson process
    };

    //Create datatype for brittle damage
    struct BrittleDamageData {
      bool printDamage;    /* Flag to print damage */

      // used for Bhasker's damage model:
      double KIc;               // Critical stress intensity factor
      double mu;                // Crack face friction coefficient
      double phi;               // angle between crack normal and max comp stress
      double cgamma;            // Exponent for crack growth speed
      double alpha;             // Multiplier for max crack velocity
      double criticalDamage;    // Damage level to start granular flow or mark as failed
      double maxDamage;         // Damage level to stop damage evolution
      // Use plane strain assumption for SCM calculation
      bool usePlaneStrain;

      // Control damage evolution timestepping:
      double maxDamageInc; /* Maximum damage increment in a time step */
      bool   useDamageTimeStep; // Control the global timestep with the damage timestep
      bool   useOldStress;      // Compute the damage based on the stress from the previous timestep
      double dt_increaseFactor;
      bool   incInitialDamage; // Include the initial flaw as a part of the damage level
      bool   doFlawInteraction; // do the ellipse calculation for flaw interactions
      bool   useNonlocalDamage; // Use averaged damage from cell interpolation
    };

    struct granularPlasticityData {
      double timeConstant;      // Time constant for viscoplastic update (0 is rate independent)
      double JGP_loc;           // Value of JGP to trigure localized particles

      // Parameters that define the Granular plastic yeild surface:
      double A;                 // Damaged scale parameter
      double B;                 // Damaged hydrostatic tensile strength
      int    yeildSurfaceType;  // 1 for cone with hemispherical cap, 2 for parabola

      // P-alpha compaction model parameters:
      double Pc;                // Pressure (+ compression) for full compaction
      double alpha_e;           // Distension corrisponding to elastic compaction pressure
      double Pe;                // Pressure required to start compaction at J^{GP}=\alpha_e
    };

    // const VarLabel* pFailureStressOrStrainLabel;
    const VarLabel* pLocalizedLabel;
    const VarLabel* pDamageLabel;
    const VarLabel* pDeformRateLabel;
    // const VarLabel* pFailureStressOrStrainLabel_preReloc;
    const VarLabel* pLocalizedLabel_preReloc;
    const VarLabel* pDamageLabel_preReloc;
    const VarLabel* pDeformRateLabel_preReloc;
    // const VarLabel* bBeBarLabel;          // Are these used or do i just need bElBarLabel
    // const VarLabel* bBeBarLabel_preReloc; // Are these used
    const VarLabel* pEnergyLabel;
    const VarLabel* pEnergyLabel_preReloc;
      
    // Plasticity Requirements //
    /////////////////////////////
    const VarLabel* pPlasticStrain_label;
    const VarLabel* pPlasticStrain_label_preReloc;
    const VarLabel* pPlasticEnergy_label;
    const VarLabel* pPlasticEnergy_label_preReloc;

    // Granular Plasticity Variables:
    const VarLabel* pGPJLabel;
    const VarLabel* pGPJLabel_preReloc;
    const VarLabel* pGP_plasticStrainLabel;
    const VarLabel* pGP_plasticStrainLabel_preReloc;
    const VarLabel* pGP_plasticEnergyLabel;
    const VarLabel* pGP_plasticEnergyLabel_preReloc;

    // Nonlocal damage:
    const VarLabel* gDamage_Label;

    MPMEquationOfState* d_eos;

      
  protected:
    // Flags indicating if damage and/or plasticity should be used
    bool d_useDamage;
    bool d_usePlasticity;
    bool d_useGranularPlasticity;
      
    // Basic Requirements //
    ////////////////////////
    CMData d_initialData;
    // bool d_useModifiedEOS; 
    int d_8or27;
      
    // Damage Requirments //
    ////////////////////////
    // FailureStressOrStrainData d_epsf;
    BrittleDamageData d_brittle_damage;
    flawDistributionData d_flawDistData;

    granularPlasticityData d_GPData;
      
    // Erosion algorithms
    bool d_setStressToZero; /* set stress tensor to zero*/
    bool d_allowNoTension;  /* retain compressive mean stress after failue*/
    bool d_allowNoShear;    /* retain mean stress after failure - no deviatoric stress */
                            /* i.e., no deviatoric stress */
      
  ///////////////
  // Functions //
  ///////////////
  private:
    // Prevent copying of this class
    // copy constructor
    //TongeRamesh(const TongeRamesh &cm);
    TongeRamesh& operator=(const TongeRamesh &cm);

    double calc_yeildFunc_g_gs_gp
    (const double sigma_s, const double sigma_p,
     double &gs, double &gp);

    void calc_returnDir(const double sigma_p, const double sigma_s,
                        const double JGP,
                        double &M_p, double &M_s
                        );
      
    // Plasticity requirements
    //friend const TypeDescription* fun_getTypeDescriptiont(StateData*);

  public:
    // constructors
    TongeRamesh(ProblemSpecP& ps, MPMFlags* flag);
    TongeRamesh(ProblemSpecP& ps, MPMFlags* flag, bool plas, bool dam);
    TongeRamesh(const TongeRamesh* cm);

    // specifcy what to output from the constitutive model to an .xml file
    virtual void outputProblemSpec(ProblemSpecP& ps, bool output_cm_tag = true);
    
    // clone
    TongeRamesh* clone();
      
    // destructor
    virtual ~TongeRamesh();
    
    
    // Initialization Functions //
    //////////////////////////////
    // virtual void allocateCMDataAdd(DataWarehouse* new_dw,
    //                                ParticleSubset* subset,
    //                                map<const VarLabel*, ParticleVariableBase*>* newState,
    //                                ParticleSubset* delset,
    //                                DataWarehouse* old_dw);
    
    // virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
    //                                        const PatchSet* patch, 
    //                                        MPMLabel* lb) const;
    
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
    virtual void computeStressTensor(const PatchSubset* patches,
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
    // void getFailureStressOrStrainData(ProblemSpecP& ps);

    void getBrittleDamageData(ProblemSpecP& ps);

    void getFlawDistributionData(ProblemSpecP& ps);

    void getGranularPlasticityData(ProblemSpecP& ps);    

    // void setFailureStressOrStrainData(const TongeRamesh* cm);      

    void setBrittleDamageData(const TongeRamesh* cm);

    void setFlawDistributionData(const TongeRamesh* cm);
    
    void setGranularPlasticityData(const TongeRamesh* cm);
      
    void initializeLocalMPMLabels();
      
    void setErosionAlgorithm();
      
    void setErosionAlgorithm(const TongeRamesh* cm);

    double calculateDamageGrowth(Matrix3 &stress,
                                 vector<double> &N,
                                 vector<double> &s,
                                 vector<double> &old_L,
                                 const double currentDamage,
                                 vector<double> &new_L,
                                 vector<double> &new_Ldot,
                                 const double dt,
                                 const int Localized,
                                 const PlasticityState *state
                                 );

    // void computeIncStress_Bhasker(const double matrixStress[3], double incStress[3],
    //                               const double currentDamage, const PlasticityState *state);

    // void computeIncStress_Junwei(const double matrixStress[3], double incStress[3],
    //                               const double currentDamage, const PlasticityState *state);

    void computeIncStress(const double matrixStress[3], double incStress[3],
                          const double wingDamage, const double parentDamage,
                          const PlasticityState *state);

    
    double calculateShearPrefactor(const double currentDamage, const PlasticityState *state
                               );

    double calculateBulkPrefactor(const double currentDamage, const PlasticityState *state,
                                  const double J = 1.0
                              );

    double computePressure(const MPMMaterial *mat, const Matrix3 &F, const Matrix3 &dF,
                           const PlasticityState *state, const double delT,
                           const double currentDamage=0);
    
    double pareto_PDF(const double s, const double s_min, const double s_max,
                      const double a)
    {
      return a * pow(s_min, a) * pow(s, -a-1.0) / (1-pow(s_min/s_max,a));
    }

    double pareto_CDF(const double s, const double s_min, const double s_max,
                      const double a)
    {
      return (1-pow(s_min/s,a))/(1-pow(s_min/s_max,a));
    }

    double pareto_invCDF(const double u, const double s_min, const double s_max,
                      const double a)
    {
      // return (s_max*s_min)/(pow( (1-u)*pow(s_max,a) + u*pow(s_min,a), 1.0/a));
      double HL_ratio = s_max/s_min;
      double HL_ratio_a = pow(HL_ratio, a);
      return s_max/pow(HL_ratio_a-u*(HL_ratio_a-1),1.0/a);
    }

    double pareto_FirstMoment(const double s_min, const double s_max, const double a)
    {
      // For a bounded Pareto distribution, the pdf for a sub section of the distribution
      // has an identical form to the parent distribution. The only difference is that the
      // max and minimum values are changed.
      if( fabs(a-1.0)>1e-8 ){
        return pow(s_min,a)/(1-pow(s_min/s_max,a))
          * a/(a-1) * (pow(s_min,1-a) - pow(s_max,1-a));
      } else {
        // Special case of a=1:
        // return s_min*(log(s_max)-log(s_min))/(1-s_min/s_max);
        return s_min*log(s_max/s_min)/(1-s_min/s_max);
      }
    }

    double pareto_ThirdMoment(const double s_min, const double s_max, const double a)
    {
      if( fabs(a-3.0)>1e-8 ){
        return pow(s_min,a)/(1-pow(s_min/s_max,a))
          * a/(a-3) * (pow(s_min,3-a) - pow(s_max,3-a));
      } else {
        // Special case of a=3:
        return 3.0* pow(s_min,3) * log(s_max/s_min)/(1-pow(s_min/s_max,3));
      }
    }

    double pareto_SixthMoment(const double s_min, const double s_max, const double a)
    {
      if( fabs(a-6.0)>1e-8 ){
        return pow(s_min,a)/(1-pow(s_min/s_max,a))
          * a/(a-6.0) * (pow(s_min,6.0-a) - pow(s_max,6.0-a));
      } else {
        // Special case of a=6:
        return 6.0* pow(s_min,6) * log(s_max/s_min) /(1-pow(s_min/s_max,6));
      }
    }

    double pareto_variance(const double s_min, const double s_max, const double a)
    {
      double secondMoment;
      if( fabs(a-2.0)>1e-8 ){
        double firstMoment(pareto_FirstMoment(s_min,s_max,a));
        secondMoment =  pow(s_min,a)/(1-pow(s_min/s_max,a))
          * a/(a-2) * (pow(s_min,2-a) - pow(s_max,2-a));
        secondMoment -= (firstMoment*firstMoment);
      } else {
        // Special case of a=2:
        secondMoment = s_min*s_min*log(s_max/s_min)/(1-(s_min*s_min)/(s_max*s_max));
      }
      return secondMoment;
    }

    int poisson_sample(MTRand &randGen, const double lambda){
      double L(exp(-lambda)), p(1.0);
      int k(0);
      do{
        ++k;
        p *= randGen.rand();
      } while (p>L);
      return k-1;
    }

    double logNormal_PDF(const double s, const double sig, const double mu)
    {
      double temp(log(s)-mu);
      temp *= temp;
      temp /= -(2.0*sig*sig);
      temp = exp(temp);
      temp /= (s*sig*sqrt(2*M_PI));
      return temp;
    }

    
  protected:
    // compute stress at each particle in the patch
    void computeStressTensorImplicit(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);
    
    // Modify the stress for failed particles and
    // check if the particle has failed
    void checkStabilityAndDoErosion(const Matrix3& defGrad,
                                    const double& currentDamage,
                                    const int& pLocalized,
                                    int& pLocalized_new,
                                    Matrix3& pStress,
                                    const long64 particleID,
                                    const PlasticityState *state);


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
      


#endif  //  __TONGE_RAMESH_CONSTITUTIVE_MODEL_H__

