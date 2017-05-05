/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 * Copyright (c) 2013-2014 Callaghan Innovation, New Zealand
 * Copyright (c) 2015-2016 Parresia Research Limited, New Zealand
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

#ifndef __PARTIALLY_SATURATED_ARENA_YIELD_CONDITION_MODEL_H__
#define __PARTIALLY_SATURATED_ARENA_YIELD_CONDITION_MODEL_H__


#include <CCA/Components/MPM/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/YieldCondition.h>
#include <CCA/Components/MPM/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/ModelState_Arena.h>
#include <CCA/Components/MPM/ConstitutiveModel/ArenaSoilBanerjeeBrannon/WeibParameters.h>

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Grid/Task.h>

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/VarLabel.h>

#include <vector>

namespace Vaango {

  /*! 
    \class  YieldCond_Arena
    \brief  The Partally saturated Arena3 yield condition
  */

  class YieldCond_Arena : public YieldCondition {
  
  public:
    
    // Constants
    static const double sqrt_two;
    static const double sqrt_three;
    static const double one_sqrt_three;

  private:

    /**
     *  These parameters are used in the actual computation
     */
    struct ModelParameters {
      double a1;
      double a2;
      double a3;
      double a4;
      double a1_failed;
      double a2_failed;
      double a3_failed;
      double a4_failed;
    };

    /**
     *  These are the parameters that are read from the input file
     */
    struct YieldFunctionParameters {
      double PEAKI1;
      double FSLOPE;
      double STREN;
      double YSLOPE;
      double PEAKI1_failed;
      double FSLOPE_failed;
      double STREN_failed;
      double YSLOPE_failed;
    };

    struct NonAssociatvityParameters {
      double BETA;
    };

    struct CapParameters {
      double CR;
    };

    struct RateParameters {
      double T1;
      double T2;
    };

    struct LocalParameters {
      double PEAKI1;
      double FSLOPE;
      double STREN;
      double YSLOPE;
      double BETA;
      double CR;
      double a1;
      double a2;
      double a3;
      double a4;
    };

    ModelParameters           d_modelParam;
    YieldFunctionParameters   d_yieldParam;
    NonAssociatvityParameters d_nonAssocParam;
    CapParameters             d_capParam;
    RateParameters            d_rateParam;
    LocalParameters           d_local;

    void checkInputParameters();
    void computeModelParameters(double fac);
    std::vector<double> computeModelParameters(const double& PEAKI1,
                                               const double& FSLOPE,
                                               const double& STREN,
                                               const double& YSLOPE);

    // Prevent copying of this class
    // copy constructor
    //YieldCond_Arena(const YieldCond_Arena &);
    YieldCond_Arena& operator=(const YieldCond_Arena &);

  public:

    //! Constructor
    /*! Creates a YieldCond_Arena function object */
    YieldCond_Arena(Uintah::ProblemSpecP& ps);
    YieldCond_Arena(const YieldCond_Arena* cm);
         
    //! Destructor 
    ~YieldCond_Arena();

    virtual void outputProblemSpec(Uintah::ProblemSpecP& ps);
         
    /*! Get parameters */
    std::map<std::string, double> getParameters() const {
      std::map<std::string, double> params;
      params["PEAKI1"] = d_yieldParam.PEAKI1;
      params["FSLOPE"] = d_yieldParam.FSLOPE;
      params["STREN"]  = d_yieldParam.STREN;
      params["YSLOPE"] = d_yieldParam.YSLOPE;
      params["PEAKI1_failed"] = d_yieldParam.PEAKI1_failed;
      params["FSLOPE_failed"] = d_yieldParam.FSLOPE_failed;
      params["STREN_failed"]  = d_yieldParam.STREN_failed;
      params["YSLOPE_failed"] = d_yieldParam.YSLOPE_failed;
      params["BETA"]   = d_nonAssocParam.BETA;
      params["CR"]     = d_capParam.CR;
      params["T1"]     = d_rateParam.T1;
      params["T2"]     = d_rateParam.T2;
      params["a1"]     = d_modelParam.a1;
      params["a2"]     = d_modelParam.a2;
      params["a3"]     = d_modelParam.a3;
      params["a4"]     = d_modelParam.a4;
      params["a1_failed"]     = d_modelParam.a1_failed;
      params["a2_failed"]     = d_modelParam.a2_failed;
      params["a3_failed"]     = d_modelParam.a3_failed;
      params["a4_failed"]     = d_modelParam.a4_failed;
      //std::cout << "Yield condition parameters are: " << std::endl;
      //for (auto param : params) {
      //  std::cout << "\t \t" << param.first << " " << param.second << std::endl;
      //}
      return params;
    }

    //--------------------------------------------------------------
    // Compute value of yield function
    //--------------------------------------------------------------
    double evalYieldCondition(const ModelStateBase* state);
    double evalYieldConditionMax(const ModelStateBase* state);

    //--------------------------------------------------------------
    // Compute df/dp  where p = volumetric stress = 1/3 Tr(sigma)
    //--------------------------------------------------------------
    double computeVolStressDerivOfYieldFunction(const ModelStateBase* state);

    //--------------------------------------------------------------
    // Compute df/dq  where q = sqrt(3 J_2), J_2 = 2nd invariant deviatoric stress
    //--------------------------------------------------------------
    double computeDevStressDerivOfYieldFunction(const ModelStateBase* state);


    /**
     * Function: getInternalPoint
     *
     * Purpose: Get a point that is inside the yield surface
     *
     * Inputs:
     *  state = state at the current time
     *
     * Returns:
     *   I1 = value of tr(stress) at a point inside the yield surface
     */
    double getInternalPoint(const ModelStateBase* state_old,
                            const ModelStateBase* state_trial);

    /**
     * Function: getClosestPoint
     *
     * Purpose: Get the point on the yield surface that is closest to a given point (2D)
     *
     * Inputs:
     *  state = current state
     *  px = x-coordinate of point
     *  py = y-coordinate of point
     *
     * Outputs:
     *  cpx = x-coordinate of closest point on yield surface
     *  cpy = y-coordinate of closest point
     *
     * Returns:
     *   true - if the closest point can be found
     *   false - otherwise
     */
    bool getClosestPoint(const ModelStateBase* state,
                         const double& px, const double& py,
                         double& cpx, double& cpy);

    //================================================================================
    // Other options below.
    //================================================================================

    // Evaluate the yield function.
    double evalYieldCondition(const double p,
                              const double q,
                              const double dummy0,
                              const double dummy1,
                              double& dummy2);

    // Evaluate yield condition (s = deviatoric stress = sigDev
    //                           p = state->pressure
    //                           p_c = state->yieldStress)
    double evalYieldCondition(const Uintah::Matrix3& sigDev,
                              const ModelStateBase* state);

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Evaluate the derivative of the yield function \f$(\Phi)\f$
      with respect to \f$\sigma_{ij}\f$.
    */
    /////////////////////////////////////////////////////////////////////////
    void evalDerivOfYieldFunction(const Uintah::Matrix3& stress,
                                  const double dummy1,
                                  const double dummy2,
                                  Uintah::Matrix3& derivative);

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Evaluate the derivative of the yield function \f$(\Phi)\f$
      with respect to \f$s_{ij}\f$.

      This is for the associated flow rule with \f$s_{ij}\f$ being
      the deviatoric stress.
    */
    /////////////////////////////////////////////////////////////////////////
    void evalDevDerivOfYieldFunction(const Uintah::Matrix3& stress,
                                     const double dummy1,
                                     const double dummy2,
                                     Uintah::Matrix3& derivative);

    /*! Derivative with respect to the Cauchy stress (\f$\sigma \f$)*/
    void eval_df_dsigma(const Uintah::Matrix3& xi,
                        const ModelStateBase* state,
                        Uintah::Matrix3& df_dsigma);

    /*! Derivative with respect to the \f$xi\f$ where \f$\xi = s - \beta \f$  
      where \f$s\f$ is deviatoric part of Cauchy stress and 
      \f$\beta\f$ is the backstress */
    void eval_df_dxi(const Uintah::Matrix3& xi,
                     const ModelStateBase* state,
                     Uintah::Matrix3& df_xi);

    /* Derivative with respect to \f$ s \f$ and \f$ \beta \f$ */
    void eval_df_ds_df_dbeta(const Uintah::Matrix3& xi,
                             const ModelStateBase* state,
                             Uintah::Matrix3& df_ds,
                             Uintah::Matrix3& df_dbeta);

    /*! Derivative with respect to the plastic strain (\f$\epsilon^p \f$)*/
    double eval_df_dep(const Uintah::Matrix3& xi,
                       const double& d_sigy_dep,
                       const ModelStateBase* state);

    /*! Derivative with respect to the porosity (\f$\epsilon^p \f$)*/
    double eval_df_dphi(const Uintah::Matrix3& xi,
                        const ModelStateBase* state);

    /*! Compute h_alpha  where \f$d/dt(ep) = d/dt(gamma)~h_{\alpha}\f$ */
    double eval_h_alpha(const Uintah::Matrix3& xi,
                        const ModelStateBase* state);

    /*! Compute h_phi  where \f$d/dt(phi) = d/dt(gamma)~h_{\phi}\f$ */
    double eval_h_phi(const Uintah::Matrix3& xi,
                      const double& factorA,
                      const ModelStateBase* state);

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the elastic-plastic tangent modulus.
    */
    /////////////////////////////////////////////////////////////////////////
    void computeElasPlasTangentModulus(const Uintah::TangentModulusTensor& Ce,
                                       const Uintah::Matrix3& sigma, 
                                       double sigY,
                                       double dsigYdep,
                                       double porosity,
                                       double voidNuclFac,
                                       Uintah::TangentModulusTensor& Cep);

    /////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate the factor \f$h_1\f$ for plastic strain

      \f[
      h_1 = \frac{\sigma : f_{\sigma}}{\sigma_Y}
      \f]
      
      \return factor 
    */
    /////////////////////////////////////////////////////////////////////////
    inline double computePlasticStrainFactor(double sigma_f_sigma,
                                             double sigma_Y);

    /////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the continuum elasto-plastic tangent modulus
      assuming associated flow rule.

      \f[
      C_{ep} = C_{e} - \frac{(C_e:f_{\sigma})\otimes(f_{\sigma}:C_e)}
      {-f_q.h_q + f_{\sigma}:C_e:f_{\sigma}}
      \f]
      
      \return TangentModulusTensor \f$ C_{ep} \f$.
    */
    /////////////////////////////////////////////////////////////////////////
    void computeTangentModulus(const Uintah::TangentModulusTensor& Ce,
                               const Uintah::Matrix3& f_sigma, 
                               double f_q1, 
                               double h_q1,
                               Uintah::TangentModulusTensor& Cep);

  public:

    // Parameter variability VarLabels
    const Uintah::VarLabel*   pPEAKI1Label;
    const Uintah::VarLabel*   pPEAKI1Label_preReloc;
    const Uintah::VarLabel*   pFSLOPELabel;
    const Uintah::VarLabel*   pFSLOPELabel_preReloc;
    const Uintah::VarLabel*   pSTRENLabel;
    const Uintah::VarLabel*   pSTRENLabel_preReloc;
    const Uintah::VarLabel*   pYSLOPELabel;
    const Uintah::VarLabel*   pYSLOPELabel_preReloc;

    const Uintah::VarLabel*   pBETALabel;
    const Uintah::VarLabel*   pBETALabel_preReloc;

    const Uintah::VarLabel*   pCRLabel;
    const Uintah::VarLabel*   pCRLabel_preReloc;

    const Uintah::VarLabel*   pT1Label;
    const Uintah::VarLabel*   pT1Label_preReloc;
    const Uintah::VarLabel*   pT2Label;
    const Uintah::VarLabel*   pT2Label_preReloc;

    // Return the yield condition parameter labels
    std::vector<const Uintah::VarLabel*> getLabels() const
      {
        std::vector<const Uintah::VarLabel*> labels;
        labels.push_back(pPEAKI1Label);
        labels.push_back(pPEAKI1Label_preReloc); 

        labels.push_back(pFSLOPELabel);
        labels.push_back(pFSLOPELabel_preReloc); 

        labels.push_back(pSTRENLabel);
        labels.push_back(pSTRENLabel_preReloc); 

        labels.push_back(pYSLOPELabel);
        labels.push_back(pYSLOPELabel_preReloc); 

        labels.push_back(pBETALabel);
        labels.push_back(pBETALabel_preReloc);

        labels.push_back(pT1Label);
        labels.push_back(pT1Label_preReloc);
    
        labels.push_back(pT2Label);
        labels.push_back(pT2Label_preReloc);


        return labels;
      }

    // Add particle state for these labels
    void addParticleState(std::vector<const Uintah::VarLabel*>& from,
                          std::vector<const Uintah::VarLabel*>& to) 
      {
        from.push_back(pPEAKI1Label);
        from.push_back(pFSLOPELabel);
        from.push_back(pSTRENLabel);
        from.push_back(pYSLOPELabel);
        from.push_back(pBETALabel);
        from.push_back(pCRLabel);
        from.push_back(pT1Label);
        from.push_back(pT2Label);

        to.push_back(pPEAKI1Label_preReloc);
        to.push_back(pFSLOPELabel_preReloc);
        to.push_back(pSTRENLabel_preReloc);
        to.push_back(pYSLOPELabel_preReloc);
        to.push_back(pBETALabel_preReloc);
        to.push_back(pCRLabel_preReloc);
        to.push_back(pT1Label_preReloc);
        to.push_back(pT2Label_preReloc);
      }

    /**
     * Initialize local VarLabels that are used for setting the parameter variability
     */
    void initializeLocalMPMLabels() 
      {
        pPEAKI1Label          = Uintah::VarLabel::create("p.ArenaPEAKI1",
                                                 Uintah::ParticleVariable<double>::getTypeDescription());
        pPEAKI1Label_preReloc = Uintah::VarLabel::create("p.ArenaPEAKI1+",
                                                 Uintah::ParticleVariable<double>::getTypeDescription());
        pFSLOPELabel          = Uintah::VarLabel::create("p.ArenaFSLOPE",
                                                 Uintah::ParticleVariable<double>::getTypeDescription());
        pFSLOPELabel_preReloc = Uintah::VarLabel::create("p.ArenaFSLOPE+",
                                                 Uintah::ParticleVariable<double>::getTypeDescription());
        pSTRENLabel          = Uintah::VarLabel::create("p.ArenaSTREN",
                                                Uintah::ParticleVariable<double>::getTypeDescription());
        pSTRENLabel_preReloc = Uintah::VarLabel::create("p.ArenaSTREN+",
                                                Uintah::ParticleVariable<double>::getTypeDescription());
        pYSLOPELabel          = Uintah::VarLabel::create("p.ArenaYSLOPE",
                                                 Uintah::ParticleVariable<double>::getTypeDescription());
        pYSLOPELabel_preReloc = Uintah::VarLabel::create("p.ArenaYSLOPE+",
                                                 Uintah::ParticleVariable<double>::getTypeDescription());

        pBETALabel          = Uintah::VarLabel::create("p.ArenaBETA",
                                               Uintah::ParticleVariable<double>::getTypeDescription());
        pBETALabel_preReloc = Uintah::VarLabel::create("p.ArenaBETA+",
                                               Uintah::ParticleVariable<double>::getTypeDescription());

        pCRLabel          = Uintah::VarLabel::create("p.ArenaCR",
                                             Uintah::ParticleVariable<double>::getTypeDescription());
        pCRLabel_preReloc = Uintah::VarLabel::create("p.ArenaCR+",
                                             Uintah::ParticleVariable<double>::getTypeDescription());
    
        pT1Label          = Uintah::VarLabel::create("p.ArenaT1",
                                             Uintah::ParticleVariable<double>::getTypeDescription());
        pT1Label_preReloc = Uintah::VarLabel::create("p.ArenaT1+",
                                             Uintah::ParticleVariable<double>::getTypeDescription());
        pT2Label          = Uintah::VarLabel::create("p.ArenaT2",
                                             Uintah::ParticleVariable<double>::getTypeDescription());
        pT2Label_preReloc = Uintah::VarLabel::create("p.ArenaT2+",
                                             Uintah::ParticleVariable<double>::getTypeDescription());
      }

    /**
     * Set up task graph for initialization
     */
    void addInitialComputesAndRequires(Uintah::Task* task,
                                       const Uintah::MPMMaterial* matl,
                                       const Uintah::PatchSet* patch) const 
      {
        const Uintah::MaterialSubset* matlset = matl->thisMaterial(); 
        task->computes(pPEAKI1Label,    matlset);
        task->computes(pFSLOPELabel,    matlset);
        task->computes(pSTRENLabel,     matlset);
        task->computes(pYSLOPELabel,    matlset);
        task->computes(pBETALabel,      matlset);
        task->computes(pCRLabel,        matlset);
        task->computes(pT1Label,        matlset);
        task->computes(pT2Label,        matlset);
      }

    /**
     *  Actually initialize the variability parameters
     */
    void initializeLocalVariables(const Uintah::Patch* patch,
                                  Uintah::ParticleSubset* pset,
                                  Uintah::DataWarehouse* new_dw,
                                  Uintah::constParticleVariable<double>& pVolume)
      {
        Uintah::ParticleVariable<double> pPEAKI1, pFSLOPE, pSTREN, pYSLOPE; 
        Uintah::ParticleVariable<double> pBETA, pCR, pT1, pT2;

        new_dw->allocateAndPut(pPEAKI1,    pPEAKI1Label,    pset);
        new_dw->allocateAndPut(pFSLOPE,    pFSLOPELabel,    pset);
        new_dw->allocateAndPut(pSTREN,     pSTRENLabel,     pset);
        new_dw->allocateAndPut(pYSLOPE,    pYSLOPELabel,    pset);
        new_dw->allocateAndPut(pBETA,      pBETALabel,      pset);
        new_dw->allocateAndPut(pCR,        pCRLabel,        pset);
        new_dw->allocateAndPut(pT1,        pT1Label,        pset);
        new_dw->allocateAndPut(pT2,        pT2Label,        pset);

        // Default (constant) initialization
        for (auto iter = pset->begin(); iter != pset->end(); iter++) {
          Uintah::particleIndex idx = *iter;
          pPEAKI1[idx] = d_yieldParam.PEAKI1;
          pFSLOPE[idx] = d_yieldParam.FSLOPE;
          pSTREN[idx] = d_yieldParam.STREN;
          pYSLOPE[idx] = d_yieldParam.YSLOPE;
          pBETA[idx] = d_nonAssocParam.BETA;
          pCR[idx] = d_capParam.CR;
          pT1[idx] = d_rateParam.T1;
          pT2[idx] = d_rateParam.T2;
        }

        // Weibull initialization if parameters are allowed to vary
        d_weibull_PEAKI1.assignWeibullVariability(patch, pset, pVolume, pPEAKI1);
        d_weibull_FSLOPE.assignWeibullVariability(patch, pset, pVolume, pFSLOPE);
        d_weibull_STREN.assignWeibullVariability(patch, pset, pVolume, pSTREN);
        d_weibull_YSLOPE.assignWeibullVariability(patch, pset, pVolume, pYSLOPE);
        d_weibull_BETA.assignWeibullVariability(patch, pset, pVolume, pBETA);
        d_weibull_CR.assignWeibullVariability(patch, pset, pVolume, pCR);
        d_weibull_T1.assignWeibullVariability(patch, pset, pVolume, pT1);
        d_weibull_T2.assignWeibullVariability(patch, pset, pVolume, pT2);
      }

    /**
     * Set up task graph for parameter copying to new datawarehouse
     */
    void addComputesAndRequires(Uintah::Task* task,
                                const Uintah::MPMMaterial* matl,
                                const Uintah::PatchSet* patches) const 
      {
        const Uintah::MaterialSubset* matlset = matl->thisMaterial(); 
        task->requires(Uintah::Task::OldDW, pPEAKI1Label,    matlset, Uintah::Ghost::None);
        task->requires(Uintah::Task::OldDW, pFSLOPELabel,    matlset, Uintah::Ghost::None);
        task->requires(Uintah::Task::OldDW, pSTRENLabel,     matlset, Uintah::Ghost::None);
        task->requires(Uintah::Task::OldDW, pYSLOPELabel,    matlset, Uintah::Ghost::None);
        task->requires(Uintah::Task::OldDW, pBETALabel,      matlset, Uintah::Ghost::None);
        task->requires(Uintah::Task::OldDW, pCRLabel,        matlset, Uintah::Ghost::None);
        task->requires(Uintah::Task::OldDW, pT1Label,        matlset, Uintah::Ghost::None);
        task->requires(Uintah::Task::OldDW, pT2Label,        matlset, Uintah::Ghost::None);

        task->computes(pPEAKI1Label_preReloc,    matlset);
        task->computes(pFSLOPELabel_preReloc,    matlset);
        task->computes(pSTRENLabel_preReloc,     matlset);
        task->computes(pYSLOPELabel_preReloc,    matlset);
        task->computes(pBETALabel_preReloc,      matlset);
        task->computes(pCRLabel_preReloc,        matlset);
        task->computes(pT1Label_preReloc,        matlset);
        task->computes(pT2Label_preReloc,        matlset);
      }

    /**
     *  Copy the variability parameters from old_dw to new_dw
     */
    void copyLocalVariables(Uintah::ParticleSubset* pset,
                            Uintah::DataWarehouse* old_dw,
                            Uintah::DataWarehouse* new_dw) 
      {
        Uintah::constParticleVariable<double> pPEAKI1_old, pFSLOPE_old, pSTREN_old, pYSLOPE_old; 
        Uintah::constParticleVariable<double> pBETA_old, pCR_old, pT1_old, pT2_old;
        old_dw->get(pPEAKI1_old, pPEAKI1Label,    pset);
        old_dw->get(pFSLOPE_old, pFSLOPELabel,    pset);
        old_dw->get(pSTREN_old,  pSTRENLabel,     pset);
        old_dw->get(pYSLOPE_old, pYSLOPELabel,    pset);
        old_dw->get(pBETA_old,   pBETALabel,      pset);
        old_dw->get(pCR_old,     pCRLabel,        pset);
        old_dw->get(pT1_old,     pT1Label,        pset);
        old_dw->get(pT2_old,     pT2Label,        pset);

        Uintah::ParticleVariable<double> pPEAKI1_new, pFSLOPE_new, pSTREN_new, pYSLOPE_new; 
        Uintah::ParticleVariable<double> pBETA_new, pCR_new, pT1_new, pT2_new;
        new_dw->allocateAndPut(pPEAKI1_new, pPEAKI1Label_preReloc,    pset);
        new_dw->allocateAndPut(pFSLOPE_new, pFSLOPELabel_preReloc,    pset);
        new_dw->allocateAndPut(pSTREN_new,  pSTRENLabel_preReloc,          pset);
        new_dw->allocateAndPut(pYSLOPE_new, pYSLOPELabel_preReloc,    pset);
        new_dw->allocateAndPut(pBETA_new,   pBETALabel_preReloc,      pset);
        new_dw->allocateAndPut(pCR_new,     pCRLabel_preReloc,        pset);
        new_dw->allocateAndPut(pT1_new,     pT1Label_preReloc,        pset);
        new_dw->allocateAndPut(pT2_new,     pT2Label_preReloc,        pset);

        for (auto iter = pset->begin(); iter != pset->end(); iter++) {
          Uintah::particleIndex idx = *iter;
          pPEAKI1_new[idx]    = pPEAKI1_old[idx];
          pFSLOPE_new[idx]    = pFSLOPE_old[idx];
          pSTREN_new[idx]     = pSTREN_old[idx];
          pYSLOPE_new[idx]    = pYSLOPE_old[idx];
          pBETA_new[idx]      = pBETA_old[idx];
          pCR_new[idx]        = pCR_old[idx];
          pT1_new[idx]        = pT1_old[idx];
          pT2_new[idx]        = pT2_old[idx];
        }
      }

    std::vector<std::string> getLocalVariableLabels() const
      {
        std::vector<std::string> pYieldParamLabels;
        pYieldParamLabels.emplace_back("PEAKI1");
        pYieldParamLabels.emplace_back("FSLOPE");
        pYieldParamLabels.emplace_back("STREN");
        pYieldParamLabels.emplace_back("YSLOPE");
        pYieldParamLabels.emplace_back("BETA");
        pYieldParamLabels.emplace_back("CR");
        pYieldParamLabels.emplace_back("T1");
        pYieldParamLabels.emplace_back("T2");
    
        return pYieldParamLabels;
      }

    std::vector<Uintah::constParticleVariable<double> >
    getLocalVariables(Uintah::ParticleSubset* pset,
                      Uintah::DataWarehouse* old_dw)
      {
        Uintah::constParticleVariable<double> pPEAKI1, pFSLOPE, pSTREN, pYSLOPE; 
        Uintah::constParticleVariable<double> pBETA, pCR, pT1, pT2;
        old_dw->get(pPEAKI1, pPEAKI1Label,    pset);
        old_dw->get(pFSLOPE, pFSLOPELabel,    pset);
        old_dw->get(pSTREN,  pSTRENLabel,     pset);
        old_dw->get(pYSLOPE, pYSLOPELabel,    pset);
        old_dw->get(pBETA,   pBETALabel,      pset);
        old_dw->get(pCR,     pCRLabel,        pset);
        old_dw->get(pT1,     pT1Label,        pset);
        old_dw->get(pT2,     pT2Label,        pset);

        std::vector<Uintah::constParticleVariable<double> > pYieldParams;
        pYieldParams.emplace_back(pPEAKI1);
        pYieldParams.emplace_back(pFSLOPE);
        pYieldParams.emplace_back(pSTREN);
        pYieldParams.emplace_back(pYSLOPE);
        pYieldParams.emplace_back(pBETA);
        pYieldParams.emplace_back(pCR);
        pYieldParams.emplace_back(pT1);
        pYieldParams.emplace_back(pT2);
    
        return pYieldParams;
      }

    /**
     *  This is used to scale the yield parameters 
     */
    virtual
    void updateLocalVariables(Uintah::ParticleSubset* pset,
                              Uintah::DataWarehouse* old_dw,
                              Uintah::DataWarehouse* new_dw,
                              Uintah::constParticleVariable<double>& pCoherence_old,
                              const Uintah::ParticleVariable<double>& pCoherence_new);

  private :

    Uintah::WeibParameters d_weibull_PEAKI1;
    Uintah::WeibParameters d_weibull_FSLOPE;
    Uintah::WeibParameters d_weibull_STREN;
    Uintah::WeibParameters d_weibull_YSLOPE;

    Uintah::WeibParameters d_weibull_BETA;

    Uintah::WeibParameters d_weibull_CR;

    Uintah::WeibParameters d_weibull_T1;
    Uintah::WeibParameters d_weibull_T2;

    /* Find the closest point */
    void getClosestPointAlgebraicBisect(const ModelState_Arena* state,
                                        const Uintah::Point& z_r_pt, 
                                        Uintah::Point& z_r_closest); 
    void getClosestPointGeometricBisect(const ModelState_Arena* state,
                                        const Uintah::Point& z_r_pt, 
                                        Uintah::Point& z_r_closest); 

    /* Get the closest point on the yield surface */
    void findClosestPoint(const Uintah::Point& p, 
                          const std::vector<Uintah::Point>& poly,
                          Uintah::Point& min_p);

    /* Get the points on the yield surface */
    void getYieldSurfacePointsAll_RprimeZ(const double& X_eff,
                                          const double& kappa,
                                          const double& sqrtKG,
                                          const double& I1eff_min,
                                          const double& I1eff_max,
                                          const int& num_points,
                                          std::vector<Uintah::Point>& polyline);
    void getYieldSurfacePointsSegment_RprimeZ(const double& X_eff,
                                              const double& kappa,
                                              const double& sqrtKG,
                                              const Uintah::Point& start_point,
                                              const Uintah::Point& end_point,
                                              const int& num_points,
                                              std::vector<Uintah::Point>& polyline);

    /* linspace function */
    void linspace(const double& start, const double& end, const int& num,
                  std::vector<double>& linspaced);
    std::vector<double> linspace(double start, double end, int num);

    /*! Compute a vector of z_eff, r' values given a range of I1_eff values */
    void computeZeff_and_RPrime(const double& X_eff,
                                const double& kappa,
                                const double& sqrtKG,
                                const double& I1eff_min,
                                const double& I1eff_max,
                                const int& num_points,
                                std::vector<Uintah::Point>& z_r_vec);

    /* Get closest segments */
    void getClosestSegments(const Uintah::Point& pt, 
                            const std::vector<Uintah::Point>& poly,
                            std::vector<Uintah::Point>& segments);
  };

} // End namespace Uintah

#endif  // __PARTIALLY_SATURATED_ARENA_YIELD_CONDITION_MODEL_H__ 
