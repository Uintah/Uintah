/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef UINTAH_HOMEBREW_Component_UdaReducer_H
#define UINTAH_HOMEBREW_Component_UdaReducer_H

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <CCA/Ports/Output.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimpleMaterial.h>

#include <vector>

namespace Uintah {
  class LoadBalancerPort;

/**************************************

CLASS
   SimulationInterface
   
   Short description...

GENERAL INFORMATION

   SimulationInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Simulation_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class UdaReducer : public SimulationInterface, public UintahParallelComponent {
  public:
    UdaReducer( const ProcessorGroup * myworld, 
                const std::string    & d_udaDir );

    virtual ~UdaReducer();

    virtual void problemSetup( const ProblemSpecP     & params, 
                               const ProblemSpecP     & restart_prob_spec, 
                                     GridP            & grid, 
                                     SimulationStateP & state );

    virtual void scheduleInitialize( const LevelP     & level,
                                           SchedulerP & );
                                           
    virtual void scheduleRestartInitialize( const LevelP     & level,
                                                  SchedulerP & );

    virtual void restartInitialize() {}

    virtual void scheduleComputeStableTimestep( const LevelP &,
                                                      SchedulerP & );

    virtual void scheduleTimeAdvance( const LevelP     & level,
                                            SchedulerP & );


    virtual bool needRecompile( const double   time,
                                const double   dt,
                                const GridP  & grid );
                               
    virtual void scheduleFinalizeTimestep(const LevelP& level, 
                                          SchedulerP&){};

    // stubs
    virtual void scheduleInitialErrorEstimate  ( const LevelP& , SchedulerP&  ){};
    virtual void scheduleCoarsen               ( const LevelP& , SchedulerP&  ){};
    virtual void scheduleRefine                ( const PatchSet*, SchedulerP& ){};
    virtual void scheduleRefineInterface       ( const LevelP& , SchedulerP& , bool, bool){};

    
    double getMaxTime();
    
    double getInitialTime();

    GridP getGrid();
  //______________________________________________________________________
  //  
  private:
    UdaReducer(const UdaReducer&);
    UdaReducer& operator=(const UdaReducer&);

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,     
                    const MaterialSubset* matls,    
                    DataWarehouse* /*old_dw*/,      
                    DataWarehouse* new_dw);         

    void computeDelT(const ProcessorGroup*,
                     const PatchSubset* patches,    
                     const MaterialSubset* matls,   
                     DataWarehouse* /*old_dw*/,     
                     DataWarehouse* new_dw);        

    void sched_readDataArchive(const LevelP& level,
                               SchedulerP& sched);

    void readDataArchive(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* /*old_dw*/,
                         DataWarehouse* new_dw);

    void finalizeTimestep(const ProcessorGroup*,
                          const PatchSubset*,
                          const MaterialSubset*,
                          DataWarehouse*,
                          DataWarehouse* );

    std::string            d_udaDir;
    bool                   d_gridChanged;     

    std::vector<int>       d_timesteps;
    std::vector<int>       d_numMatls;
    std::vector<double>    d_times;
    std::vector<VarLabel*> d_savedLabels;

    GridP                  d_oldGrid;
    DataArchive          * d_dataArchive;
    Output               * d_dataArchiver;
    
    int                    d_timeIndex;

    LoadBalancerPort     * d_lb;
    const VarLabel       * delt_label;
    SimulationStateP       d_sharedState;
    SimpleMaterial       * d_oneMatl;
  };
} // End namespace Uintah
   


#endif
