/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_OnTheFlyAnalysis_spatialAvg_h
#define Packages_Uintah_CCA_Components_OnTheFlyAnalysis_spatialAvg_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/LevelP.h>

namespace Uintah{

/**************************************

CLASS
   spatialAvg

GENERAL INFORMATION

   spatialAvg.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah


KEYWORDS
   spatialAvg

DESCRIPTION
   Computes spatial averages for CCVariable<double,float,Vector>


WARNING

****************************************/
  class spatialAvg : public AnalysisModule {
  public:
    spatialAvg(const ProcessorGroup*  myworld,
               const MaterialManagerP materialManager,
               const ProblemSpecP&    module_spec);

    spatialAvg();

    virtual ~spatialAvg();

    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);
    
    virtual void outputProblemSpec( ProblemSpecP& ps){};

    virtual void scheduleInitialize(SchedulerP   & sched,
                                    const LevelP & level);
    
    virtual void scheduleRestartInitialize(SchedulerP& sched,
                                           const LevelP& level){};

    virtual void scheduleDoAnalysis(SchedulerP   & sched,
                                    const LevelP & level);

    void scheduleDoAnalysis_preReloc(SchedulerP   & sched,
                                     const LevelP & level) {};

    std::string getName(){ return "spatialAvg"; };

  private:

    //__________________________________
    //  container to hold each variable to compute stats
    struct QavgVar{
      int matl;
      VarLabel* Q_Label;
      VarLabel* avgLabel;
      VarLabel* varianceLabel;

      const Uintah::TypeDescription* subtype;

      void print(){
        const std::string name = Q_Label->getName();
        std::cout << name << " matl: " << matl << " subtype: " << subtype->getName()  <<"\n";
      };
    };

    //__________________________________
    //
    void initialize(const ProcessorGroup  *,
                    const PatchSubset     * patches,
                    const MaterialSubset  *,
                    DataWarehouse         *,
                    DataWarehouse         * new_dw);


    void doAnalysis(const ProcessorGroup  * pg,
                    const PatchSubset     * patches,
                    const MaterialSubset  *,
                    DataWarehouse         *,
                    DataWarehouse         * new_dw);

    template <class T>
    void computeAvgWrapper( DataWarehouse * old_dw,
                            DataWarehouse * new_dw,
                            const PatchSubset* patches,
                            const Patch   * patch,
                            QavgVar        & Q);
    template <class T>
    void computeAvg( DataWarehouse  * old_dw,
                     DataWarehouse  * new_dw,
                     const Patch    * patch,
                     QavgVar         & Q);

    template <class T>
    void computeTimeAverage( const Patch         * patch,
                             CellIterator        & iter,
                             constCCVariable< T >& Qvar,
                             constCCVariable< T >& Qvar_old,
                             CCVariable< T >     & Qavg,
                             const int           & timeStep);

    template <class T>
    void query( const Patch         * patch,
                constCCVariable<T>  & Qvar,
                CCVariable<T>       & Qavg,
                CCVariable<T>       & Qvariance,
                IntVector           & avgBoxCells,
                CellIterator        & iter);

    template <class T>
    void allocateAndZeroLabels( DataWarehouse * new_dw,
                                const Patch   * patch,
                                QavgVar        & Q);

    template <class T>
    void allocateAndZero( DataWarehouse  * new_dw,
                          const VarLabel * label,
                          const int        matl,
                          const Patch    * patch );

    enum Domain {EVERYWHERE, INTERIOR, BOUNDARIES};
    //__________________________________
    // global constants
    double    d_startTime  = 0;
    double    d_stopTime   = DBL_MAX;

    Domain    d_compDomain = EVERYWHERE;            // domain to compute averages
    IntVector d_monitorCell = IntVector(-9,-9,-9);  // Cell to output
    IntVector d_avgBoxCells;                        // number of cells to average over.
    std::vector< QavgVar >  d_QavgVars;

    MaterialSet          * d_matlSet = {nullptr};
    const MaterialSubset * d_matSubSet  {nullptr};
    
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
