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

#ifndef Packages_Uintah_CCA_Components_PostProcessUda_Module_h
#define Packages_Uintah_CCA_Components_PostProcessUda_Module_h

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/ComputeSet.h>

namespace Uintah {

  class DataWarehouse;
  class Material;
  class Patch;
  class Module {

  public:

    Module();
    Module(ProblemSpecP     & prob_spec,
           MaterialManagerP & materialManager,
           Output           * dataArchiver,
           DataArchive      * dataArchive );

    virtual ~Module();

    virtual void problemSetup() = 0;

    virtual void scheduleInitialize(SchedulerP &   sched,
                                    const LevelP & level) =0;

    virtual void scheduleDoAnalysis(SchedulerP &   sched,
                                    const LevelP & level) =0;

    virtual void scheduleDoAnalysis_preReloc(SchedulerP   & sched,
                                             const LevelP & level) =0;
                                             
    // populate the old_dw with variables from timestep
    enum {NOTUSED = -9};
    virtual int getTimestep_OldDW(){ return NOTUSED; };
    
    virtual std::string getName()=0;
    
    void readTimeStartStop(const ProblemSpecP & ps,
                           double & startTime,
                           double & stopTime);

    void createMatlSet(const ProblemSpecP        &  module_ps,
                       MaterialSet               * matlSet,
                       std::map<std::string,int> & Qmatls);

    template <class T>
    void allocateAndZero( DataWarehouse  * new_dw,
                          const VarLabel * label,
                          const int        matl,
                          const Patch    * patch );
    MaterialManagerP   d_materialManager;
    DataArchive      * d_dataArchive   = nullptr;
    Output           * d_dataArchiver  = nullptr;
    std::vector<double> d_udaTimes;                 // physical time pulled from uda:index.xml
    
    //__________________________________
    //
    class proc0patch0cout {
      public:
        proc0patch0cout( const int nTimesPerTimestep);
                              
        void print(const Patch * patch,
                   std::ostringstream& msg);
      private:
        int d_count             =0;        
        int d_nTimesPerTimestep =0;        
    
    };
  };
}

#endif
