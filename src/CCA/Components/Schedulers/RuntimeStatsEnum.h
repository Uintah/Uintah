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

#ifndef CCA_COMPONENTS_SCHEDULERS_RUNTIMESTATSENUMS_H
#define CCA_COMPONENTS_SCHEDULERS_RUNTIMESTATSENUMS_H


namespace Uintah {

/**************************************
      
    CLASS
      RuntimeStatsEnum
      
      
    GENERAL INFORMATION
      
      RunTimeStatsEnum.h
      
      
    KEYWORDS
      RuntimeStats Enumerations
      
    DESCRIPTION
      RuntimeStats Enums to be used by Simulation Controller.
      
      
****************************************/

  // timing statistics to test load balance
  enum RuntimeStatsEnum
  {
    // These five enumerators are used in SimulationController::ReportStats to determine the overhead time.
      CompilationTime = 0
    , RegriddingTime
    , RegriddingCompilationTime
    , RegriddingCopyDataTime
    , LoadBalancerTime
    
    // These five enumerators are used in SimulationController::ReportStats to determine task and comm overhead.
    , TaskExecTime
    , TaskLocalCommTime
    , TaskWaitCommTime
    , TaskReduceCommTime
    , TaskWaitThreadTime

    , XMLIOTime
    , OutputIOTime
    , ReductionIOTime
    , CheckpointIOTime
    , CheckpointReductionIOTime
    , TotalIOTime

    , OutputIORate
    , ReductionIORate
    , CheckpointIORate
    , CheckpointReductionIORate

    , SCIMemoryUsed
    , SCIMemoryMaxUsed
    , SCIMemoryHighwater

    , MemoryUsed
    , MemoryResident

    , NumTasks
    , NumPatches
    , NumCells
    , NumParticles

  };

} // end namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_RUNTIMESTATSENUMS_H
