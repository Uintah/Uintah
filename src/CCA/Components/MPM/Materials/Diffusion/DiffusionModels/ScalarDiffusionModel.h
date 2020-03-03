/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef UINTAH_RF_SCALARDIFFUSIONMODEL_H
#define UINTAH_RF_SCALARDIFFUSIONMODEL_H

#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>

#include <Core/Grid/Level.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <string>
#include <CCA/Components/MPM/Materials/Diffusion/ConductivityModels/ConductivityEquation.h>

namespace Uintah {

  class Task;

  enum FluxDirection{
    fd_in,
    fd_out,
    fd_transition
  };

  class ScalarDiffusionModel {
  public:
    
              ScalarDiffusionModel(ProblemSpecP     & ps,
                                   MaterialManagerP & sS,
                                   MPMFlags         * Mflag,
                                   std::string        diff_type);
    virtual  ~ScalarDiffusionModel();

    // Functions which are constant for all diffusion models.
    std::string getDiffusionType() const;

    virtual double getMaxConcentration() const;

    virtual double getMinConcentration() const;

    virtual double getConcentrationTolerance() const;

    virtual void setIncludeHydroStress(bool value);

    virtual void initializeTimeStep(const Patch         * patch,
                                    const MPMMaterial   * matl,
                                          DataWarehouse * new_dw
                                   );

    // Functions required to be implemented  for the individual model.
    virtual void addInitialComputesAndRequires(      Task         * task,
                                               const MPMMaterial  * matl,
                                               const PatchSet     * patches
                                              ) const = 0;

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to
                                 ) const = 0;


    virtual void computeFlux(const Patch          * patch,
                             const MPMMaterial    * matl,
                                   DataWarehouse  * old_dw,
                                   DataWarehouse  * new_dw
                            ) = 0;

    virtual void initializeSDMData(const Patch          * patch,
                                   const MPMMaterial    * matl,
                                         DataWarehouse  * new_dw
                                  ) = 0;

    virtual void scheduleComputeFlux(      Task         * task,
                                     const MPMMaterial  * matl,
                                                 const PatchSet     * patch
                                                ) const = 0;

    virtual void addSplitParticlesComputesAndRequires(      Task        * task,
                                                      const MPMMaterial * matl,
                                                      const PatchSet    * patches
                                                     ) const = 0;

    virtual void splitSDMSpecificParticleData(const Patch                 * patch,
                                              const int dwi,
                                              const int nDims,
                                                    ParticleVariable<int> & prefOld,
                                                    ParticleVariable<int> & pref,
                                              const unsigned int            oldNumPar,
                                              const int                     numNewPartNeeded,
                                                    DataWarehouse         * old_dw,
                                                    DataWarehouse         * new_dw
                                             ) = 0;

    virtual void outputProblemSpec(
                                   ProblemSpecP & ps,
                                   bool         output_rdm_tag = true
                                  ) const = 0;


    // Functions which have a default implementation, but may need to be
    //   overrideen.

    virtual void scheduleComputeDivergence(      Task         * task,
                                           const MPMMaterial  * matl,
                                           const PatchSet     * patch
                                          ) const;

    virtual void computeDivergence(const Patch          * patch,
                                   const MPMMaterial    * matl,
                                         DataWarehouse  * old_dw,
                                         DataWarehouse  * new_dw
                                  );

    virtual void scheduleComputeDivergence_CFI(      Task         * task,
                                               const MPMMaterial  * matl,
                                               const PatchSet     * patch
                                              ) const;

    virtual void computeDivergence_CFI(const PatchSubset    * finePatches,
                                       const MPMMaterial    * matl,
                                             DataWarehouse  * old_dw,
                                             DataWarehouse  * new_dw
                                      );

    virtual double computeStableTimeStep(double Dif,
                                         Vector dx
                                        ) const;

    virtual bool usesChemicalPotential();

    virtual void addChemPotentialComputesAndRequires(
                                                           Task         * task,
                                                     const MPMMaterial  * matl,
                                                     const PatchSet     * patches
                                                    ) const;

    virtual void calculateChemicalPotential(
                                            const PatchSubset   * patches,
                                            const MPMMaterial   * matl,
                                                  DataWarehouse * old_dw,
                                                  DataWarehouse * new_dw
                                           );

    virtual double computeDiffusivityTerm(double concentration,
                                          double pressure
                                         );

    virtual ConductivityEquation* getConductivityEquation();

  protected:

    void baseInitializeSDMData(
                               const Patch          * patch,
                               const MPMMaterial    * matl,
                                     DataWarehouse  * new_dw
                              );

    void baseOutputSDMProbSpec(
                               ProblemSpecP & ps,
                               bool         output_rdm_tag = true
                              ) const;

    MPMLabel* d_lb;
    MPMFlags* d_Mflag;
    MaterialManagerP d_materialManager;
    ConductivityEquation* d_conductivity_equation;

    int NGP, NGN;
    std::string diffusion_type;
    bool include_hydrostress;

    ScalarDiffusionModel(const ScalarDiffusionModel&);
    ScalarDiffusionModel& operator=(const ScalarDiffusionModel&);
    
    double d_D0;
    double d_MaxConcentration;
    double d_MinConcentration;
    double d_concTolerance;
    double d_InitialConcentration;
    double d_InverseMaxConcentration;

    Matrix3 d_StrainFreeExpansion;

    MaterialSubset* d_one_matl;         // matlsubset for zone of influence
  };
  
} // end namespace Uintah
#endif
