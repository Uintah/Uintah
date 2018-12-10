/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_PostProcessUda_spatioTemporalAvg_h
#define Packages_Uintah_CCA_Components_PostProcessUda_spatioTemporalAvg_h

#include <CCA/Components/PostProcessUda/Module.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/LevelP.h>

namespace Uintah{
namespace postProcess{

/**************************************

CLASS
   spatioTemporalAvg

GENERAL INFORMATION

   spatioTemporalAvg.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah


KEYWORDS
   spatioTemporalAvg

DESCRIPTION
   Computes spatial,temporal averages for CCVariable<double,float,Vector>


WARNING

****************************************/
  class spatioTemporalAvg : public Module {
  public:
    spatioTemporalAvg(ProblemSpecP    & prob_spec,
                      MaterialManagerP& materialManager,
                      Output          * dataArchiver,
                      DataArchive     * dataArchive);

    spatioTemporalAvg(); 

    virtual ~spatioTemporalAvg();

    virtual void problemSetup();

    virtual void scheduleInitialize(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleDoAnalysis(SchedulerP   & sched,
                                    const LevelP & level);

    void scheduleDoAnalysis_preReloc(SchedulerP   & sched,
                                     const LevelP & level) {};
                                     
    virtual int getTimestep_OldDW();

    std::string getName(){ return "spatioTemporalAvg"; };

  private:

    //__________________________________
    //  container to hold each variable to compute stats
    struct Qstats{
      int matl;
      VarLabel* Q_Label;
      VarLabel* avgLabel;
      VarLabel* varianceLabel;

      const Uintah::TypeDescription* subtype;

      // Code for keeping track of which timestep
      int timestep;
      bool isSet;

      void initializeTimestep(){
        timestep = 0;
        isSet    = false;
      }

      int getStart(){
        return timestep;
      }

      // only set the timestep once
      void setStart( const int me) {
        if(isSet == false){
          timestep = me;
          isSet   = true;
        }
      }

      void print(){
        const std::string name = Q_Label->getName();
        std::cout << name << " matl: " << matl << " subtype: " << subtype->getName() << " startTimestep: " << timestep <<"\n";
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
                            Qstats        & Q);
    template <class T>
    void computeAvg( DataWarehouse  * old_dw,
                     DataWarehouse  * new_dw,
                     const Patch    * patch,
                     Qstats         & Q);

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
                                Qstats        & Q);              
                          
    enum Domain {EVERYWHERE, INTERIOR, BOUNDARIES};
    //__________________________________
    // global constants
    const VarLabel* m_simulationTimeLabel;
    const VarLabel* m_timeStepLabel;

    double    d_startTime  = 0;
    double    d_stopTime   = DBL_MAX;
    bool      d_doTemporalAvg = false;
    int       d_baseTimestep  = NOTUSED;           // timestep used in computing time averages
    
    Domain    d_compDomain = EVERYWHERE;            // domain to compute averages
    IntVector d_monitorCell = IntVector(-9,-9,-9);  // Cell to output
    IntVector d_avgBoxCells;                        // number of cells to average over.
    std::vector< Qstats >  d_Qstats;

    ProblemSpecP        d_prob_spec;
    MaterialSet       * d_matlSet = nullptr;
    LoadBalancer  * d_lb      = nullptr;
  };
}
}
#endif
