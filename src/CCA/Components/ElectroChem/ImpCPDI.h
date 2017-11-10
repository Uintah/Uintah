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

#ifndef UINTAH_ELECTROCHEM_IMPCPDI_H
#define UINTAH_ELECTROCHEM_IMPCPDI_H

#include <CCA/Ports/SimulationInterface.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Components/ElectroChem/ImpECFlags.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Parallel/ProcessorGroup.h>


namespace Uintah{
/**************************************

CLASS
   ImpCPDI
   
   Short description...

GENERAL INFORMATION

   ImpCPDI.h

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class ImpCPDI : public UintahParallelComponent, public SimulationInterface{
public:
  ImpCPDI(const ProcessorGroup* myworld);
  virtual ~ImpCPDI();

  void problemSetup( const ProblemSpecP&     params,
                     const ProblemSpecP&     restart_prob_spec,
                           GridP&            grid,
                           SimulationStateP& state );

  void preGridProblemSetup( const ProblemSpecP&     params, 
                                  GridP&            grid,
                                  SimulationStateP& state );

  void outputProblemSpec( ProblemSpecP& ps );
      
  void scheduleInitialize( const LevelP&     level,
                                 SchedulerP& sched );
                                 
  void scheduleRestartInitialize( const LevelP&     level,
                                        SchedulerP& sched );

  void scheduleComputeStableTimestep( const LevelP&     level,
                                            SchedulerP& sched );
      
  void scheduleTimeAdvance(const LevelP& level, SchedulerP& sched);

private:
  ImpCPDI(const ImpCPDI&);
  ImpCPDI& operator=(const ImpCPDI&);

  SimulationStateP d_shared_state;
  MPMLabel* d_mpm_lb;
  ImpECFlags* d_impec_flags;

  MaterialSubset* d_one_matl;

  double d_next_output_time;
  double d_SMALL_NUM;
  double d_initial_dt;
  double d_stop_time;

  int d_NGP;
  int d_NGN;
  int d_num_iterations;

}; // end class ImpCPDI
} // end namespace Uintah

#endif // UINTAH_ELECTROCHEM_IMPCPDI_H
