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

#ifndef UINTAH_HOMEBREW_AMRSIMULATIONCONTROLLER_H
#define UINTAH_HOMEBREW_AMRSIMULATIONCONTROLLER_H

#include <CCA/Components/SimulationController/SimulationController.h>

namespace Uintah {

/**************************************
      
  CLASS
       AMRSimulationController
      
       Short description...
      
  GENERAL INFORMATION
      
       AMRSimulationController.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
       
             
  KEYWORDS
       Simulation_Controller
      
  DESCRIPTION
       Long description...
     
  WARNING
      
****************************************/

//! Controls the execution of an AMR Simulation
class AMRSimulationController : public SimulationController {

   public:
     AMRSimulationController( const ProcessorGroup * myworld
                            ,       bool             doAMR
                            ,       ProblemSpecP     pspec
                            );

     virtual ~AMRSimulationController(){};
     
     virtual void run();


   protected:

     AMRSimulationController( const AMRSimulationController& );

     AMRSimulationController& operator=( const AMRSimulationController& );
     
     //! Set up, compile, and execute initial timestep
     void doInitialTimestep( );

     //! Execute a timestep
     void executeTimestep( int totalFine, int tg_index );

     //! If doing AMR do the regridding
     bool doRegridding( bool initialTimestep );

     //! Asks a variety of components if one of them needs the
     //! taskgraph to recompile.
     bool needRecompile();
     
     void recompile( int totalFine );

     //! recursively schedule refinement, coarsening, and time advances for finer levels,
     //! compensating for time refinement. Builds one taskgraph
     void subCycleCompile( int startDW
                         , int dwStride
                         , int numLevel
                         , int step
                         );
     
     //! recursively executes taskgraphs, as several were executed.
     //! Similar to subCycleCompile, except that this executes the
     //! recursive taskgraphs, and compile builds one taskgraph (to
     //! exsecute once) recursively.
     void subCycleExecute( int startDW
                         , int dwStride
                         , int numLevel
                         , bool rootCycle
                         );
     
     void scheduleComputeStableTimestep();
     
     void reduceSysVar( const ProcessorGroup *
                      , const PatchSubset    * patches
                      , const MaterialSubset * /*matls*/
                      ,       DataWarehouse  * /*old_dw*/
                      ,       DataWarehouse  * new_dw
                      );

     // Optional flag for scrubbing, defaulted to true.
     bool d_scrubDataWarehouse{true};

     // Barrier timers used when running and regridding.
     Timers::Simple barrierTimer;
     double barrier_times[5];
   };

} // End namespace Uintah

#endif
