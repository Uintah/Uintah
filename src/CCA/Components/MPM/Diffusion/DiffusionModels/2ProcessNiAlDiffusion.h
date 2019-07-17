/*
 * 2ProcessNiAlDiffusion.h
 *
 *  Created on: Jan 21, 2019
 *      Author: jbhooper
 */

#ifndef CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_2PROCESSNIALDIFFUSION_H_
#define CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_2PROCESSNIALDIFFUSION_H_

#include <CCA/Components/MPM/Diffusion/DiffusionModels/EAM_AlNi_Diffusion.h>
#include <CCA/Components/MPM/Diffusion/DiffusionModels/ScalarDiffusionModel.h>
#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/FunctionInterpolator.h>

namespace Uintah
{

  class NiAl2Process: public ScalarDiffusionModel
  {

    public:
      NiAl2Process(
                          ProblemSpecP      & ps
                  ,       SimulationStateP  & simState
                  ,       MPMFlags          * mpmFlags
                  ,       std::string         diff_type
                  );
     ~NiAl2Process();

      // Required interfaces for ScalarDiffusionModel
      void virtual addInitialComputesAndRequires(
                                                        Task        * task
                                                , const MPMMaterial * matl
                                                , const PatchSet    * patches
                                                ) const;

      void virtual addParticleState(
                                           std::vector<const VarLabel*> & from
                                   ,       std::vector<const VarLabel*> & to
                                   ) const;

      void    virtual computeFlux(
                                    const Patch         * patch
                                 ,  const MPMMaterial   * matl
                                 ,        DataWarehouse * oldDW
                                 ,        DataWarehouse * newDW
                                 );

      void    virtual initializeSDMData(
                                         const  Patch         * patch
                                       , const  MPMMaterial   * matl
                                       ,        DataWarehouse * newDW
                                       );

      void    virtual scheduleComputeFlux(
                                                  Task        * task
                                         ,  const MPMMaterial * matl
                                         ,  const PatchSet    * patches
                                         ) const;

      void    virtual addSplitParticlesComputesAndRequires(
                                                                   Task        * task
                                                          , const  MPMMaterial * matl
                                                          , const  PatchSet    * patches
                                                          ) const;

      void    virtual splitSDMSpecificParticleData(
                                                    const Patch                 * patch
                                                  , const int                     dwi
                                                  , const int                     nDims
                                                  ,       ParticleVariable<int> & pRefOld
                                                  ,       ParticleVariable<int> & pRef
                                                  , const unsigned int            oldNumPart
                                                  , const int                     numNewPartNeeded
                                                  ,       DataWarehouse         * oldDW
                                                  ,       DataWarehouse         * newDW
                                                  );

      void    virtual outputProblemSpec(ProblemSpecP  & ps
                                       ,bool            output_rdm_tag = true
                                       ) const;

      // Concentration here is a function of temperature, so we need to
      //   override the default.
      double  virtual getMaxConstClamp(const double & Temp) const;

      void    virtual scheduleTrackConcentrationThreshold(      Task        * task
                                                         ,const MPMMaterial * matl
                                                         ,const PatchSet    * patchSet
                                                         ) const;

      void    virtual trackConcentrationThreshold(const Patch         * patch
                                                 ,const MPMMaterial   * matl
                                                 ,      DataWarehouse * old_dw
                                                 ,      DataWarehouse * new_dw    );

      // We probably need to override these for this model to have better
      //   control.

//      void    virtual scheduleComputeDivergence(        Task        * task
//                                               , const  MPMMaterial * matl
//                                               , const  PatchSet    * patch
//                                               ) const;
//
//      void    virtual computeDivergence(  const Patch         * patch
//                                       ,  const MPMMaterial   * matl
//                                       ,        DataWarehouse * old_dw
//                                       ,        DataWarehouse * new_dw
//                                       );
//
//      void    virtual scheduleComputeDivergence_CFI(       Task        * task
//                                                   , const MPMMaterial * matl
//                                                   , const PatchSet    * patches
//                                                   ) const;
//
//      void    virtual computeDivergence_CFI(const PatchSubset   * finePatches
//                                           ,const MPMMaterial   * matl
//                                           ,      DataWarehouse * old_dw
//                                           ,      DataWarehouse * new_dw
//                                           );

//      double  virtual computeStableTimeStep(        double  diffusivity
//                                           ,        Vector  dx
//                                           ) const ;

//      Matrix3 virtual getStressFreeExpansionMatrix() const;

    private:
      double  maxConcentrationAtTemperature(double            Temperature
                                           ,EAM_AlNi_Region   phaseRegion) const;

            bool                    f_isConcNormalized;
            double                  m_multiplier;
            Matrix3                 m_latticeMisfit;
            FunctionInterpolator  * m_phaseInterpolator;
            double                  m_D0Liquid, m_D0Solid, m_D0LowT;
            bool                    m_lowTOldModel;
            double                  m_lowT_Al;
            double                  m_lowT_Ni;
      const VarLabel              * m_globalMinNiConc;
      const VarLabel              * m_globalMinAlConc;
      const VarLabel              * m_pRegionType;
      const VarLabel              * m_pRegionType_preReloc;

  };
}


#endif /* CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_2PROCESSNIALDIFFUSION_H_ */
