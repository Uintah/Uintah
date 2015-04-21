/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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


#ifndef ICE_SM_H
#define ICE_SM_H

#include <CCA/Components/ICE_sm/Advection/Advector.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Geometry/Vector.h>

#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/Utils.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Math/UintahMiscMath.h>

#include <vector>
#include <string>

namespace Uintah {
  using namespace SCIRun;


    class ICE_sm : public UintahParallelComponent, public SimulationInterface {
    public:
      ICE_sm(const ProcessorGroup* myworld, const bool doAMR = false);
      virtual ~ICE_sm();

      virtual bool restartableTimesteps();

      virtual double recomputeTimestep(double current_dt);

      virtual void problemSetup(const ProblemSpecP& params,
                                const ProblemSpecP& restart_prob_spec,
                                GridP& grid, SimulationStateP&);

      virtual void outputProblemSpec(ProblemSpecP& ps);


      virtual void scheduleInitialize(const LevelP& level,
                                      SchedulerP&);

      virtual void scheduleRestartInitialize(const LevelP& level,
                                                 SchedulerP& sched);

      virtual void restartInitialize();

      virtual void scheduleComputeStableTimestep(const LevelP&,
                                                SchedulerP&);

      virtual void scheduleTimeAdvance( const LevelP& level,
                                        SchedulerP&);

      void sched_ComputePressure(SchedulerP&,
                                 const PatchSet*,
                                 const MaterialSet*);

      void sched_ComputeVel_FC(SchedulerP&,
                               const PatchSet*,
                               const MaterialSet*);


      void sched_ComputeDelPressAndUpdatePressCC(SchedulerP&,
                                                 const PatchSet*,
                                                 const MaterialSet*);

      void sched_ComputePressFC(SchedulerP&,
                                const PatchSet*,
                                const MaterialSet*);

      void sched_ComputeThermoTransportProperties(SchedulerP&,
                                                  const LevelP& level,
                                                  const MaterialSet*);

      void sched_VelTau_CC( SchedulerP&,
                            const PatchSet*,
                            const MaterialSet* );

      void sched_ViscousShearStress( SchedulerP&,
                                     const PatchSet*,
                                     const MaterialSet*);


      void sched_AccumulateMomentumSourceSinks(SchedulerP&,
                                               const PatchSet*,
                                               const MaterialSet*);

      void sched_AccumulateEnergySourceSinks(SchedulerP&,
                                             const PatchSet*,
                                             const MaterialSet*);

      void sched_ComputeLagrangianValues(SchedulerP&,
                                         const PatchSet*,
                                         const MaterialSet*);

      void sched_AdvectAndAdvanceInTime(SchedulerP&,
                                        const PatchSet*,
                                        const MaterialSet*);

      void sched_ConservedtoPrimitive_Vars(SchedulerP& sched,
                                           const PatchSet* patch_set,
                                           const MaterialSet* ice_matls);

      void sched_TestConservation(SchedulerP&,
                                  const PatchSet*,
                                  const MaterialSet*);

    public:

      void actuallyInitialize(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse*,
                              DataWarehouse* new_dw);


      void actuallyComputeStableTimestep(const ProcessorGroup*,
                                         const PatchSubset* patch,
                                         const MaterialSubset* matls,
                                         DataWarehouse*,
                                         DataWarehouse*);

      void computeEquilPressure_1_matl(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

      void computeVel_FC(const ProcessorGroup*,
                         const PatchSubset*,
                         const MaterialSubset*,
                         DataWarehouse*,
                         DataWarehouse*);

      template<class T>
      void computeVelFace(int dir, CellIterator it,
                          IntVector adj_offset,double dx,
                          double delT, double gravity,
                          constCCVariable<double>& rho_CC,
                          constCCVariable<double>& sp_vol_CC,
                          constCCVariable<Vector>& vel_CC,
                          constCCVariable<double>& press_CC,
                          T& vel_FC,
                          T& gradP_FC );

      void computeDelPressAndUpdatePressCC(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* matls,
                                           DataWarehouse*,
                                           DataWarehouse*);

      void computePressFC(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse*,
                          DataWarehouse*);

      template<class T>
      void computePressFace(CellIterator it,
                            IntVector adj_offset,
                            constCCVariable<double>& sum_rho,
                            constCCVariable<double>& press_CC,
                            T& press_FC);

      void computeThermoTransportProperties(const ProcessorGroup*,
                                            const PatchSubset* patches,
                                            const MaterialSubset* ice_matls,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw);

      void VelTau_CC(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* ice_matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);

      void computeVelTau_CCFace( const Patch* patch,
                                 const Patch::FaceType face,
                                 constCCVariable<Vector>& vel_CC,
                                 CCVariable<Vector>& velTau_CC );

      void viscousShearStress(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ice_matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

      void accumulateMomentumSourceSinks(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse*,
                                         DataWarehouse*);

      void accumulateEnergySourceSinks(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse*,
                                       DataWarehouse*);


      void computeLagrangianValues(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse*);

      void advectAndAdvanceInTime(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse*,
                                  DataWarehouse*);

      void conservedtoPrimitive_Vars(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

//__________________________________
//   O T H E R
      void TestConservation(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse*);

      IntVector upwindCell_X(const IntVector& c,
                             const double& var,
                             double is_logical_R_face );

      IntVector upwindCell_Y(const IntVector& c,
                             const double& var,
                             double is_logical_R_face );

      IntVector upwindCell_Z(const IntVector& c,
                             const double& var,
                             double is_logical_R_face );

      Vector getGravity() const {
        return d_gravity;
      }

      // flags
      int d_matl;              // ice material index
      bool d_viscousFlow;

    private:
      ICELabel* lb;
      SimulationStateP d_sharedState;
      Output* dataArchiver;

      double d_EVIL_NUM;
      double d_SMALL_NUM;
      double d_CFL;
      Vector d_gravity;

      //__________________________________
      // Misc
      Advector* d_advector;
      int  d_OrderOfAdvection;
      bool d_useCompatibleFluxes;

      // flags for the conservation test
       struct conservationTest_flags{
        bool onOff;
        bool mass;
        bool momentum;
        bool energy;
       };
       conservationTest_flags* d_conservationTest;

      ICE_sm(const ICE_sm&);
      ICE_sm& operator=(const ICE_sm&);
  };

} // End namespace Uintah

#endif
