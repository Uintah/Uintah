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
      //
      void  problemSetup( const ProblemSpecP& prob_spec,
                          const ProblemSpecP& rmcrt_ps,
                          const GridP& grid );

      //__________________________________
      //
      void sched_radiometer( const LevelP& level,
                             SchedulerP& sched,
                             Task::WhichDW abskg_dw,
                             Task::WhichDW sigma_dw,
                             Task::WhichDW celltype_dw );

      //__________________________________
      //
      void sched_initialize_VRFlux( const LevelP& level,
                                    SchedulerP& sched );

      //__________________________________
      //
      template< class T >
      void radiometerFlux( const Patch  * patch,
                           const Level  * level,
                           DataWarehouse* new_dw,
                           MTRand& mTwister,
                           constCCVariable< T > sigmaT4OverPi,
                           constCCVariable< T > abskg,
                           constCCVariable<int> celltype,
                           const bool modifiesFlux );


      const VarLabel* d_VRFluxLabel{nullptr};       // computed radiometer flux
      const VarLabel* d_VRIntensityLabel{nullptr};  // computed Intensity

    private:

      // radiometer variables
      struct radiometer {
        double theta_rotate;
        double phi_rotate;
        double xi_rotate;
        double theta_viewAngle;
        double range;
        double solidAngle;

        int    nRays{1000};
        Point  locationsMin;
        Point  locationsMax;
      };

      std::vector<radiometer*> d_radiometers;

      inline const VarLabel* getRadiometerLabel() const {
        return d_VRFluxLabel;
      }

      //__________________________________
      //
      template < class T >
      void initialize_VRFlux( const ProcessorGroup  *,
                              const PatchSubset     * patches,
                              const MaterialSubset  * matls,
                              DataWarehouse         * old_dw,
                              DataWarehouse         * new_dw );

      //__________________________________
      //
      void getPatchSet( SchedulerP  & sched,
                        const LevelP& level,
                        std::vector<radiometer* > radiometers,
                        PatchSet    * ps);

      //__________________________________
      //
      template< class T >
      void radiometerTask( const ProcessorGroup * pc,
                          const PatchSubset     * patches,
                          const MaterialSubset  * matls,
                          DataWarehouse         * old_dw,
                          DataWarehouse         * new_dw,
                          Task::WhichDW   which_abskg_dw,
                          Task::WhichDW   which_sigmaT4_dw,
                          Task::WhichDW   which_celltype_dw );

      //__________________________________
      //
      void rayDirection_VR( MTRand& mTwister,
                            const IntVector& origin,
                            const int iRay,
                            const radiometer* VR,
                            Vector& directionVector,
                            double& cosVRTheta );

  }; // class Radiometer

} // namespace Uintah

#endif
