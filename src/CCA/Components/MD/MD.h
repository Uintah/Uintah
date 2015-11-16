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

#ifndef UINTAH_MD_H
#define UINTAH_MD_H

#include <Core/Parallel/UintahParallelComponent.h>

#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/Nonbonded/Nonbonded.h>
#include <CCA/Components/MD/Electrostatics/Electrostatics.h>
#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/Integrators/Integrator.h>
#include <CCA/Components/MD/CoordinateSystems/CoordinateSystem.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/SimulationInterface.h>
#include <CCA/Ports/SwitchingCriteria.h>

#include <vector>
#include <fstream>
#include <iomanip>

namespace Uintah {

  static DebugStream md_dbg(             "MDDebug",                 false);
  static DebugStream md_cout(            "MDCout",                  false);
  static DebugStream particleDebug(      "MDParticleVariableDebug", false);
  static DebugStream electrostaticDebug( "MDElectrostaticDebug",    false);
  static DebugStream mdFlowDebug(        "MDLogicFlowDebug",        false);

  class SimpleMaterial;
  class SPME;


  typedef long64 MDAtomIndex;
  typedef std::complex<double> dblcomplex;

  /**
   *  @class MD
   *  @ingroup MD
   *  @author Alan Humphrey and Justin Hooper
   *  @date   December, 2012
   *
   *  @brief
   *
   *  @param
   */
  class MD : public UintahParallelComponent, public SimulationInterface {

    public:

      /**
       * @brief
       * @param
       * @return
       */
      enum IntegratorType {
        Explicit, Implicit,
      };

      /**
       * @brief
       * @param
       */
      MD(const ProcessorGroup* myworld);

      /**
       * @brief
       * @param
       */
      virtual ~MD();


      virtual void preGridProblemSetup(const ProblemSpecP&      params,
                                             GridP&             grid,
                                             SimulationStateP&  simState);
      /**
       * @brief
       * @param
       * @return
       */
      virtual void problemSetup( const ProblemSpecP&     params,
                                 const ProblemSpecP&     restart_prob_spec,
                                       GridP&            grid,
                                       SimulationStateP& simState );

      /**
       * @brief
       * @param
       * @return
       */
      virtual void scheduleInitialize( const LevelP&     level,
                                             SchedulerP& sched );

      /**
       * @brief
       * @param
       * @return
       */
      virtual void scheduleRestartInitialize( const LevelP&     level,
                                                    SchedulerP& sched );

      /**
       * @brief
       * @param
       * @return
       */
      virtual void scheduleComputeStableTimestep( const LevelP& level,
                                                        SchedulerP& );

      /**
       * @brief
       * @param
       * @return
       */
      virtual void scheduleTimeAdvance( const LevelP& level,
                                              SchedulerP& );


    protected:

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleOutputStatistics(       SchedulerP&   sched,
                                     const PatchSet*     patches,
                                     const MaterialSet*  atomTypes,
                                     const LevelP&       level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleKineticCalculations(       SchedulerP&    sched,
                                        const PatchSet*      perProcPatches,
                                        const MaterialSet*   atomTypes,
                                        const LevelP&        level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleNonbondedInitialize(       SchedulerP&  sched,
                                        const PatchSet*    perProcPatches,
                                        const MaterialSet* matls,
                                        const LevelP&      level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleNonbondedSetup(       SchedulerP&  sched,
                                   const PatchSet*    patches,
                                   const MaterialSet* matls,
                                   const LevelP&      level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleNonbondedCalculate(       SchedulerP&  sched,
                                       const PatchSet*    patches,
                                       const MaterialSet* matls,
                                       const LevelP&      level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleNonbondedFinalize(       SchedulerP&  sched,
                                      const PatchSet*    patches,
                                      const MaterialSet* matls,
                                      const LevelP&      level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleElectrostaticsInitialize(       SchedulerP&  sched,
                                             const PatchSet*    perProcPatches,
                                             const MaterialSet* matls,
                                             const LevelP&      level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleElectrostaticsSetup(       SchedulerP&  sched,
                                        const PatchSet*    patches,
                                        const MaterialSet* matls,
                                        const LevelP&      level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleElectrostaticsCalculate(       SchedulerP&  sched,
                                            const PatchSet*    perProcPatches,
                                            const MaterialSet* matls,
                                            const LevelP&      level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleElectrostaticsFinalize(       SchedulerP&  sched,
                                           const PatchSet*    patches,
                                           const MaterialSet* matls,
                                           const LevelP&      level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleIntegratorInitialize(       SchedulerP&   sched,
                                         const PatchSet*     patches,
                                         const MaterialSet*  matls,
                                         const LevelP&       level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleIntegratorSetup(       SchedulerP&   sched,
                                    const PatchSet*     patches,
                                    const MaterialSet*  matls,
                                    const LevelP&       level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleIntegratorCalculate(       SchedulerP&   sched,
                                        const PatchSet*     patches,
                                        const MaterialSet*  matls,
                                        const LevelP&       level );

      /**
       * @brief
       * @param
       * @return
       */
      void scheduleIntegratorFinalize(        SchedulerP&   sched,
                                        const PatchSet*     patches,
                                        const MaterialSet*  matls,
                                        const LevelP&       level );

      /**
       * @brief
       * @param
       * @return
       */
      virtual void scheduleSwitchTest( const LevelP&     level,
                                             SchedulerP& sched );

      /**
       * @brief
       * @param
       * @return
       */
      virtual void switchInitialize(const LevelP&     level,
                                          SchedulerP&  /*sched*/ );


    private:

      /**
       * @brief
       * @param
       * @return
       */
      void initialize( const ProcessorGroup* pg,
                       const PatchSubset*    patches,
                       const MaterialSubset* matls,
                             DataWarehouse*  old_dw,
                             DataWarehouse*  new_dw );

      /**
       * @brief
       * @param
       * @return
       */
      void outputStatistics( const ProcessorGroup* pg,
                             const PatchSubset*    patches,
                             const MaterialSubset* atomTypes,
                                   DataWarehouse*  oldDW,
                                   DataWarehouse*  newDW );

      /**
       * @brief
       * @param
       * @return
       */
      void computeStableTimestep( const ProcessorGroup* pg,
                                  const PatchSubset*    patches,
                                  const MaterialSubset* matls,
                                        DataWarehouse*  old_dw,
                                        DataWarehouse*  new_dw );

      /**
       * @brief
       * @param
       * @return
       */
      void calculateKineticEnergy( const ProcessorGroup* pg,
                                   const PatchSubset*    patches,
                                   const MaterialSubset* atomTypes,
                                         DataWarehouse*  oldDW,
                                         DataWarehouse*  newDW );
      /**
       * @brief
       * @param
       * @return
       */
      void nonbondedInitialize( const ProcessorGroup* pg,
                                const PatchSubset*    patches,
                                const MaterialSubset* matls,
                                      DataWarehouse*  oldDW,
                                      DataWarehouse*  newDW );

      /**
       * @brief
       * @param
       * @return
       */
      void nonbondedSetup( const ProcessorGroup* pg,
                           const PatchSubset*    patches,
                           const MaterialSubset* matls,
                                 DataWarehouse*  oldDW,
                                 DataWarehouse*  newDW );

      /**
       * @brief
       * @param
       * @return
       */
      void nonbondedCalculate( const ProcessorGroup* pg,
                               const PatchSubset*    patches,
                               const MaterialSubset* matls,
                                     DataWarehouse*  oldDW,
                                     DataWarehouse*  newDW );

      /**
       * @brief
       * @param
       * @return
       */
      void nonbondedFinalize( const ProcessorGroup*  pg,
                              const PatchSubset*     patches,
                              const MaterialSubset*  matls,
                                    DataWarehouse*   oldDW,
                                    DataWarehouse*   newDW );

      /**
       * @brief
       * @param
       * @return
       */
      void electrostaticsInitialize( const ProcessorGroup* pg,
                                     const PatchSubset*    patches,
                                     const MaterialSubset* matls,
                                           DataWarehouse*  oldDW,
                                           DataWarehouse*  newDW );

      /**
       * @brief
       * @param
       * @return
       */
      void electrostaticsSetup( const ProcessorGroup* pg,
                                const PatchSubset*    patches,
                                const MaterialSubset* matls,
                                      DataWarehouse*  oldDW,
                                      DataWarehouse*  newDW );

      /**
       * @brief
       * @param
       * @return
       */
      void electrostaticsCalculate( const ProcessorGroup* pg,
                                    const PatchSubset*    perprocPatches,
                                    const MaterialSubset* matls,
                                         DataWarehouse*   parentOldDW,
                                         DataWarehouse*   parentNewDW,
                                    const LevelP          level );

      /**
       * @brief
       * @param
       * @return
       */
      void electrostaticsFinalize( const ProcessorGroup* pg,
                                   const PatchSubset*    patches,
                                   const MaterialSubset* matls,
                                         DataWarehouse*  oldDW,
                                         DataWarehouse*  newDW) ;

      /**
       * @brief
       * @param
       * @return
       */
      void newUpdatePosition( const ProcessorGroup*  pg,
                              const PatchSubset*     patches,
                              const MaterialSubset*  atomTypes,
                                    DataWarehouse*   oldDW,
                                    DataWarehouse*   newDW );

      /**
       * @brief
       * @param
       * @return
       */
      void integratorInitialize( const ProcessorGroup*   pg,
                                 const PatchSubset*      patches,
                                 const MaterialSubset*   atomTypes,
                                       DataWarehouse*    oldDW,
                                       DataWarehouse*    newDW );

      /**
       * @brief
       * @param
       * @return
       */
      void integratorSetup(     const ProcessorGroup*   pg,
                                const PatchSubset*      patches,
                                const MaterialSubset*   atomTypes,
                                      DataWarehouse*    oldDW,
                                      DataWarehouse*    newDW );

      /**
       * @brief
       * @param
       * @return
       */
      void integratorCalculate( const ProcessorGroup*   pg,
                                const PatchSubset*      patches,
                                const MaterialSubset*   atomTypes,
                                      DataWarehouse*    oldDW,
                                      DataWarehouse*    newDW );

      /**
       * @brief
       * @param
       * @return
       */
      void integratorFinalize(  const ProcessorGroup*   pg,
                                const PatchSubset*      patches,
                                const MaterialSubset*   atomTypes,
                                      DataWarehouse*    oldDW,
                                      DataWarehouse*    newDW );

      /**
       * @brief
       * @param
       * @return
       */
       void createBasePermanentParticleState();

       /**
        * @brief
        * @param
        * @return
        */
      inline bool containsAtom( const IntVector& l,
                                const IntVector& h,
                                const IntVector& p ) const
      {
        return ((p.x() >= l.x() && p.x() < h.x()) &&
                (p.y() >= l.y() && p.y() < h.y()) &&
                (p.z() >= l.z() && p.z() < h.z()));
      }


    private:

      // Member pointers inherited from Uintah
      Output*            d_dataArchiver;     //!< Handle to the Uintah data archiver
      MDLabel*           d_label;            //!< Variable labels for the per-particle Uintah MD variables
      SimulationStateP   d_sharedState;      //!< Shared simulation state (global)
      ProblemSpecP       d_problemSpec;      //!< Problem spec since we need to parse our coordinates, and it's either storing this pointer or the whole parsed coordinate set
      ProblemSpecP       d_restartSpec;

      // Member pointers constructed specifically for this component
      Electrostatics*    d_electrostatics;   //!< The simulation Electrostatics model instance
      Nonbonded*         d_nonbonded;              //!< The simulation Nonbonded instance
      Integrator*        d_integrator;       //!< MD style integrator object

      Forcefield*        d_forcefield;       //!< Currently employed MD forcefield
      MDSystem*          d_system;           //!< The global MD system
      CoordinateSystem*  d_coordinate;       //!< Interface to abstract coordinate system

      SwitchingCriteria* d_switchCriteria;   //!< Used for switching between MD and MPM - switchFlag


      //  Does this need to exist here?  Can it be stuffed in the electrostatics object?
      SchedulerP        d_electrostaticSubscheduler;    //!< Subscheduler for SPME::calculate() convergence loop
      bool              d_recompileSubscheduler;        //!< Whether or not the subscheduler taskgraph needs recompilation

      double            delt;                           //!< Simulation delta T

      std::vector<const VarLabel*> d_particleState;            //!< Atom (particle) state prior to relocation
      std::vector<const VarLabel*> d_particleState_preReloc;   //!< For atom (particle) relocation

      // copy constructor and assignment operator (privatized on purpose)
      MD(const MD&);
      MD& operator=(const MD&);

      double d_referenceEnergy;
      double d_baseTimeStep;
      bool   d_referenceStored;

      bool   d_firstIntegration;
      bool   d_secondIntegration;
      double d_KineticBase;
      double d_PotentialBase;

      int    d_minGridLevel; // Only do MD on this grid level
      int    d_maxGridLevel; // Only do MD on this grid level
  };

}

#endif
