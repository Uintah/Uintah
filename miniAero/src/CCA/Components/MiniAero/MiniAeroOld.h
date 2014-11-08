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


#ifndef __MINI_AERO_H__
#define __MINI_AERO_H__

#include <CCA/Ports/Output.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/Utils.h>
#include <CCA/Components/MiniAero/MiniAeroLabel.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Math/UintahMiscMath.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Geometry/Vector.h>
#include <vector>
#include <string>
#include <sci_defs/hypre_defs.h>

#define MAX_MATLS 16

namespace Uintah {
  using namespace SCIRun;


    class MiniAero : public UintahParallelComponent, public SimulationInterface {
    public:
      MiniAero(const ProcessorGroup* myworld);
      virtual ~MiniAero();

      virtual bool restartableTimesteps();

      virtual double recomputeTimestep(double current_dt);

      virtual void problemSetup(const ProblemSpecP& params,
                                const ProblemSpecP& restart_prob_spec,
                                GridP& grid, SimulationStateP&);

      virtual void outputProblemSpec(ProblemSpecP& ps);


      virtual void scheduleInitialize(const LevelP& level,
                                      SchedulerP&);

      virtual void restartInitialize();

      virtual void scheduleComputeStableTimestep(const LevelP&,
                                                SchedulerP&);

      virtual void scheduleTimeAdvance( const LevelP& level,
                                        SchedulerP&);

      virtual void scheduleFinalizeTimestep(const LevelP& level, SchedulerP&);


      void scheduleComputePressure(SchedulerP&,
                                   const PatchSet*,
                                   const MaterialSubset*,
                                   const MaterialSet*);

      void scheduleComputeVel_FC(SchedulerP&,
                                      const PatchSet*,
                                      const MaterialSubset*,
                                      const MaterialSubset*,
                                      const MaterialSet*);


      void scheduleComputeDelPressAndUpdatePressCC(SchedulerP&,
                                             const PatchSet*,
                                             const MaterialSubset*,
                                             const MaterialSubset*,
                                             const MaterialSet*);

      void scheduleAddExchangeContributionToFCVel(SchedulerP&,
                                            const PatchSet*,
                                            const MaterialSubset*,
                                            const MaterialSet*,
                                            bool);

      void scheduleComputePressFC(SchedulerP&,
                              const PatchSet*,
                              const MaterialSubset*,
                              const MaterialSet*);

      void scheduleComputeThermoTransportProperties(SchedulerP&,
                                                    const LevelP& level,
                                                    const MaterialSet*);

      void scheduleVelTau_CC( SchedulerP&,
                              const PatchSet*,
                              const MaterialSet* );

      void scheduleViscousShearStress( SchedulerP&,
                                       const PatchSet*,
                                       const MaterialSet*);


      void scheduleAccumulateMomentumSourceSinks(SchedulerP&,
                                            const PatchSet*,
                                            const MaterialSubset*,
                                            const MaterialSubset*,
                                            const MaterialSet*);

      void scheduleAccumulateEnergySourceSinks(SchedulerP&,
                                            const PatchSet*,
                                            const MaterialSubset*,
                                            const MaterialSubset*,
                                            const MaterialSet*);

      void scheduleComputeLagrangianValues(SchedulerP&,
                                           const PatchSet*,
                                           const MaterialSet*);

      void scheduleComputeLagrangianSpecificVolume(SchedulerP&,
                                                   const PatchSet*,
                                                   const MaterialSubset*,
                                                   const MaterialSubset*,
                                                   const MaterialSet*);

      void scheduleAdvectAndAdvanceInTime(SchedulerP&,
                                          const PatchSet*,
                                          const MaterialSubset*,
                                          const MaterialSet*);

      void scheduleConservedtoPrimitive_Vars(SchedulerP& sched,
                                             const PatchSet* patch_set,
                                             const MaterialSubset* ice_matlsub,
                                             const MaterialSet* ice_matls,
                                             const std::string& where);

      void scheduleTestConservation(SchedulerP&,
                                    const PatchSet*,
                                    const MaterialSubset*,
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

      void updateVel_FC(const ProcessorGroup*,
                        const PatchSubset*,
                        const MaterialSubset*,
                        DataWarehouse*,
                        DataWarehouse*,
                        bool);

      template<class T> 
      void computeVelFace(int dir, CellIterator it,
                          IntVector adj_offset,double dx,                 
                          double delT, double gravity,                    
                          constCCVariable<double>& rho_CC,                
                          constCCVariable<double>& sp_vol_CC,             
                          constCCVariable<Vector>& vel_CC,                
                          constCCVariable<double>& press_CC,              
                          T& vel_FC,                                      
                          T& gradP_FC,                                    
                          bool include_acc);                              

      template<class T> 
      void updateVelFace(int dir, CellIterator it,
                         IntVector adj_offset,double dx,                  
                         double delT,                                     
                         constCCVariable<double>& sp_vol_CC,              
                         constCCVariable<double>& press_CC,               
                         T& vel_FC,                                       
                         T& grad_dp_FC);                                  

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

      void computeLagrangianSpecificVolume(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* matls,
                                           DataWarehouse*,
                                           DataWarehouse*);

      void addExchangeToMomentumAndEnergy_1matl(const ProcessorGroup*,
                                                const PatchSubset* ,
                                                const MaterialSubset*,
                                                DataWarehouse* ,
                                                DataWarehouse* );

      template< class V, class T>
      void update_q_CC(const std::string& desc,
                      CCVariable<T>& q_CC,
                      V& q_Lagrangian,
                      const CCVariable<T>& q_advected,
                      const CCVariable<double>& mass_new,
                      const CCVariable<double>& cv_new,
                      const Patch* patch);

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
      bool d_viscousFlow;

    MaterialSubset* d_press_matl;
    MaterialSet*    d_press_matlSet;

    private:
      MiniAeroLabel* lb;
      SimulationStateP d_sharedState;
      Output* dataArchiver;

      double d_EVIL_NUM;
      double d_SMALL_NUM;
      double d_CFL;
      int    d_max_iceMatl_indx;
      Vector d_gravity;


      //__________________________________
      // Misc
      Advector* d_advector;
      int  d_OrderOfAdvection;
      bool d_useCompatibleFluxes;
      bool d_clampSpecificVolume;

      // flags for the conservation test
       struct conservationTest_flags{
        bool onOff;
        bool mass;
        bool momentum;
        bool energy;
        bool exchange;
       };
       conservationTest_flags* d_conservationTest;

      MiniAero(const MiniAero&);
      MiniAero& operator=(const MiniAero&);
  };

} // End namespace Uintah

#endif
