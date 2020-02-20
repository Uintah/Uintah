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

#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Util/DebugStream.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <CCA/Ports/DataWarehouse.h>

#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>

#include <unistd.h>

namespace Uintah {
  class KelvinVoigt : public ConstitutiveModel {
    public:
      struct elastic{ // Shear (G) and bulk (K) modulus for the elastic component
        double G;
        double K;
      };

      struct viscous{ // Shear (G) and bulk (K) modulus for the
        double etaG;
        double etaK;
      };

    private:
      elastic m_elasticConstants;
      viscous m_viscousConstants;

    public:
      KelvinVoigt(ProblemSpecP & ps
                 ,MPMFlags     * mFlags);

      virtual ~KelvinVoigt();

      // Required interfaces inherited from ConstitutiveModel base class.
      void          virtual outputProblemSpec(  ProblemSpecP & ps
                                             ,  bool            output_cm_tag
                                             );

      void          virtual addInitialComputesAndRequires(        Task        * task
                                                         ,  const MPMMaterial * matl
                                                         ,  const PatchSet    * patches
                                                         ) const;

      void          virtual initializeCMData( const  Patch         * patch
                                            , const  MPMMaterial   * matl
                                            ,        DataWarehouse * new_dw
                                            );

      void          virtual addComputesAndRequires(       Task        * task
                                                  , const MPMMaterial * matl
                                                  , const PatchSet    * patches
                                                  ) const;

      void          virtual addParticleState( std::vector<const VarLabel*>  & from
                                            , std::vector<const VarLabel*>  & to
                                            );

      double        virtual computeRhoMicroCM(      double        pressure
                                             ,const double        p_ref
                                             ,const MPMMaterial * matl
                                             ,      double        Temp
                                             ,      double        rho_guess
                                             );

      void         virtual computePressEOSCM(      double        rho_m
                                             ,      double      & press_eos
                                             ,      double        p_ref
                                             ,      double      & dp_drho
                                             ,      double      & ss_new
                                             ,const MPMMaterial * matl
                                             ,      double        temperature
                                             );

      double        virtual getCompressibility();

      // Other necessary but not required functions:

      void          virtual computeStressTensor(const PatchSubset   * patches
                                               ,const MPMMaterial   * matl
                                               ,      DataWarehouse * old_dw
                                               ,      DataWarehouse * new_dw);

      void          virtual computeStableTimestep(const Patch         * patch
                                                 ,const MPMMaterial   * matl
                                                 ,      DataWarehouse * new_dw);

      KelvinVoigt*         clone();

  };
}
