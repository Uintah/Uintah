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

#ifndef SRC_CCA_COMPONENTS_MPM_MATERIALS_CONSTITUTIVEMODEL_MAXWELLWEICHERT_H_
#define SRC_CCA_COMPONENTS_MPM_MATERIALS_CONSTITUTIVEMODEL_MAXWELLWEICHERT_H_

#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Util/DebugStream.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InvalidValue.h>

#include <CCA/Ports/DataWarehouse.h>

#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>

#include <vector>
#include <unistd.h>

namespace Uintah {
  class MaxwellWeichert : public ConstitutiveModel {
    public:
                              MaxwellWeichert (ProblemSpecP & ps
                                              ,MPMFlags     * mFlags  );

                  virtual    ~MaxwellWeichert();

      // Required interfaces inherited from ConstitutiveModel base class.

      void        virtual     outputProblemSpec(  ProblemSpecP  & ps
                                               ,  bool            output_cm_tag
                                               );

      void        virtual     addInitialComputesAndRequires(        Task          * task
                                                           ,  const MPMMaterial   * matl
                                                           ,        DataWarehouse * new_dw
                                                           );

      void        virtual     carryForward( const PatchSubset   * patches
                                          , const MPMMaterial   * matl
                                          ,       DataWarehouse * old_dw
                                          ,       DataWarehouse * new_dw
                                          );

      void        virtual     initializeCMData( const Patch         * patch
                                              , const MPMMaterial   * matl
                                              ,       DataWarehouse * new_dw
                                              );

      void        virtual     addComputesAndRequires(       Task        * task
                                                    , const MPMMaterial * matl
                                                    , const PatchSet    * patches
                                                    ) const;

      void        virtual     addParticleState( std::vector<const VarLabel*>  & from
                                              , std::vector<const VarLabel*>  & to
                                              );

      double      virtual     computeRhoMicroCM(        double        pressure
                                               ,  const double        p_ref
                                               ,  const MPMMaterial * matl
                                               ,        double        Temp
                                               ,        double        rho_guess
                                               );

      void        virtual     computePressEOSCM(        double        rho_m
                                               ,        double      & press_eos
                                               ,        double        p_ref
                                               ,        double      & dp_drho
                                               ,        double      & ss_new
                                               ,  const MPMMaterial * matl
                                               ,        double        temperature
                                               );

      double      virtual     getCompressibility();

      // Other necessary but not required functions:
      void             oldComputeStressTensor(  const PatchSubset   * patches
                                                 ,  const MPMMaterial   * matl
                                                 ,        DataWarehouse * old_dw
                                                 ,        DataWarehouse * new_dw
                                                 );
      void        virtual     computeStressTensor(  const PatchSubset   * patches
                                                 ,  const MPMMaterial   * matl
                                                 ,        DataWarehouse * old_dw
                                                 ,        DataWarehouse * new_dw
                                                 );

      void        virtual     computeStableTimestep(  const Patch         * patch
                                                   ,  const MPMMaterial   * matl
                                                   ,        DataWarehouse * new_dw
                                                   );

      MaxwellWeichert*        clone();

    protected:

    private:
      double  m_KInf;       // Infinite time (terminal) bulk modulus
      double  m_GInf;       // Infinite time (terminal) shear modulus
      double  m_GammaInf;   // Normalized elastic modulus contribution (G_Inf/G_0)
      double  m_G0;         // Instantaneous shear modulus

      std::vector<std::string> m_termName; // Finite decay time term label
      std::vector<double> m_GVisco;   // Finite decay time shear moduli
      std::vector<double> m_TauVisco; // Finite decay time Tau constants
      std::vector<double> m_Gamma;    // Normalized moduli (Gamma_i = G_i/G_0 )

      // Since we want the ability to have an arbitrary length series of elements, we must
      //   build the varLabel names on the fly.  The below is the base name, and the variables
      //   will be named as:  p.stressDecay_<tag | number>, p.stressDecay_<tag | number>+ for the
      //   old/new versions.
      const std::string stressDecayTrackerBase = "p.stressDecay_";

      const VarLabel* m_pInitialStress;
      std::vector<const VarLabel*> m_stressDecayTrackers;
      const VarLabel* m_pInitialStress_preReloc;
      std::vector<const VarLabel*> m_stressDecayTrackers_preReloc;
  };
}



#endif /* SRC_CCA_COMPONENTS_MPM_MATERIALS_CONSTITUTIVEMODEL_MAXWELLWEICHERT_H_ */
