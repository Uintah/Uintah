#ifndef __VISCOSCRAM_CONSTITUTIVE_MODEL_H__
#define __VISCOSCRAM_CONSTITUTIVE_MODEL_H__

#include <Packages/Uintah/Core/Math/Matrix3.h>

namespace Uintah {
  struct ViscoScramStateData {
    Matrix3 DevStress[5];
  };   
}

#include <Core/Util/Endian.h>
namespace SCIRun {
  void swapbytes( Uintah::ViscoScramStateData& d);
} // namespace SCIRun

#include "ConstitutiveModel.h"
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class MPMLabel;
  class MPMFlags;

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ViscoScram
    \brief Light version of ViscoSCRAM
    \author Scott Bardenhagen \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2000 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////

  class ViscoScram : public ConstitutiveModel {

  public:
    struct CMData {
      double PR;
      double CoefThermExp;
      double CrackParameterA;
      double CrackPowerValue;
      double CrackMaxGrowthRate;
      double StressIntensityF;
      double CrackFriction;
      double InitialCrackRadius;
      double CrackGrowthRate;
      double G[5];
      double RTau[5];
      double Beta, Gamma;
      double DCp_DTemperature;
      int LoadCurveNumber, NumberOfPoints;
    };

    struct TimeTemperatureData {
      double T0_WLF;
      double C1_WLF;
      double C2_WLF;
    };

    typedef ViscoScramStateData StateData;
    
    const VarLabel* pVolChangeHeatRateLabel;
    const VarLabel* pViscousHeatRateLabel;
    const VarLabel* pCrackHeatRateLabel;
    const VarLabel* pCrackRadiusLabel;
    const VarLabel* pStatedataLabel;
    const VarLabel* pRandLabel;
    const VarLabel* pStrainRateLabel;

    const VarLabel* pVolChangeHeatRateLabel_preReloc;
    const VarLabel* pViscousHeatRateLabel_preReloc;
    const VarLabel* pCrackHeatRateLabel_preReloc;
    const VarLabel* pCrackRadiusLabel_preReloc;
    const VarLabel* pStatedataLabel_preReloc;
    const VarLabel* pRandLabel_preReloc;
    const VarLabel* pStrainRateLabel_preReloc;

  protected:

    friend const Uintah::TypeDescription* 
       fun_getTypeDescription(ViscoScram::StateData*);

    // Create datatype for storing model parameters
    bool d_useModifiedEOS;
    bool d_random;
    bool d_doTimeTemperature;
    bool d_useObjectiveRate;
    double d_bulk;

    CMData d_initialData;
    TimeTemperatureData d_tt;

  private:

    // Prevent assignment of this class
    ViscoScram& operator=(const ViscoScram &cm);

  public:

    // constructors
    ViscoScram(ProblemSpecP& ps, MPMLabel* lb, MPMFlags* flag);
    ViscoScram(const ViscoScram* cm);
       
    // destructor
    virtual ~ViscoScram();

    ViscoScram* clone();

    /*! Computes and requires for initialization of history variables */
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    /*! initialize  each particle's constitutive model data */
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);

    /*! compute stable timestep for this patch */
    virtual void computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);

    /*! Set up data required by and computed in computeStressTensor */
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    /*! Set up data required by and computed in computeStressTensor 
        for implicit methods */
    virtual void addComputesAndRequires(Task* ,
                                        const MPMMaterial* ,
                                        const PatchSet* ,
                                        const bool ) const
    {
    }

    /*! compute stress at each particle in the patch */
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);


    /*! carry forward CM data (computed in computeStressTensor) for RigidMPM */
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    /*! Set up data required in the particle conversion process */
    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;

    /*! Copy data from the delset to the addset in the particle 
        conversion process */
    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* subset,
        map<const VarLabel*, ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);

    /*! Add the particle data that have to be saved at the end of each 
        timestep */
    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);


    /*! Used by MPMICE for pressure equilibriation */
    virtual double computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl);

    /*! Used by MPMICE for pressure equilibriation */
    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const MPMMaterial* matl);

    /*! Used by MPMICE for pressure equilibriation */
    virtual double getCompressibility();

  };

  /*! Set up type for StateData */
  const Uintah::TypeDescription* fun_getTypeDescription(ViscoScram::StateData*);

} // End namespace Uintah
      

#endif  // __VISCOSCRAM_CONSTITUTIVE_MODEL_H__ 

