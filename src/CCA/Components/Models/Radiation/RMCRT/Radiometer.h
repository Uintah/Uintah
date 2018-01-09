/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RADIOMETER_H
#define CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RADIOMETER_H

#include <CCA/Components/Models/Radiation/RMCRT/RMCRTCommon.h>

//==========================================================================
/**
 * @class Radiometer
 * @author Todd Harman
 * @date June, 2014
 *
 *
 */

class MTRand;

namespace Uintah{

  class Radiometer : public RMCRTCommon {

    public:

      Radiometer(const TypeDescription::Type FLT_DBL );         // This class can use sigmaT4 & abskg Float or Double;

      ~Radiometer();

      //__________________________________
      //  TASKS
      /** @brief Interface to input file information */
      void  problemSetup( const ProblemSpecP& prob_spec,
                          const ProblemSpecP& rmcrt_ps,
                          const GridP& grid,
                          const bool getExtraInputs );

      /** @brief Algorithm for tracing rays from radiometer location*/
      void sched_radiometer( const LevelP& level,
                             SchedulerP& sched,
                             Task::WhichDW abskg_dw,
                             Task::WhichDW sigma_dw,
                             Task::WhichDW celltype_dw );

      void sched_initializeRadVars( const LevelP& level,
                                    SchedulerP& sched );

      template < class T >
      void initializeRadVars( const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw );

      //__________________________________
      //  FUNCTIONS
      template< class T >
      void radiometerFlux( const Patch* patch,
                           const Level* level,
                           DataWarehouse* new_dw,
                           MTRand& mTwister,
                           constCCVariable< T > sigmaT4OverPi,
                           constCCVariable< T > abskg,
                           constCCVariable<int> celltype,
                           const bool modifiesFlux );

      void getPatchSet( SchedulerP& sched,
                        const LevelP& level,
                        PatchSet* ps);
      
      inline const VarLabel* getRadiometerLabel() const {
        return d_VRFluxLabel;
      }

    private:

      // Virtual Radiometer parameters
      int    d_nRadRays{1000};                     // number of rays per radiometer used to compute radiative flux
      double d_viewAng{180.0};
      Point  d_VRLocationsMin;
      Point  d_VRLocationsMax;

      struct VR_variables {
        double thetaRot;
        double phiRot;
        double psiRot;
        double deltaTheta;
        double range;
        double sldAngl;
      };

      VR_variables d_VR;
      const VarLabel* d_VRFluxLabel{nullptr};      // computed radiometer flux

      //__________________________________
      //
      template< class T >
      void radiometer( const ProcessorGroup* pc,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw,
                       Task::WhichDW which_abskg_dw,
                       Task::WhichDW whichd_sigmaT4_dw,
                       Task::WhichDW which_celltype_dw );

      //__________________________________
      //
      void rayDirection_VR( MTRand& mTwister,
                            const IntVector& origin,
                            const int iRay,
                            VR_variables& VR,
                            Vector& directionVector,
                            double& cosVRTheta );

  }; // class Radiometer

} // namespace Uintah

#endif // CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RADIOMETER_H
