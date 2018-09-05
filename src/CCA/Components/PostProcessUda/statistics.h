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


#ifndef Packages_Uintah_CCA_Components_PostProcessUda_statistics_h
#define Packages_Uintah_CCA_Components_PostProcessUda_statistics_h

#include <CCA/Components/PostProcessUda/Module.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/LevelP.h>

namespace Uintah{
namespace postProcess{

/**************************************

CLASS
   statistics

GENERAL INFORMATION

   statistics.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah


KEYWORDS
   statistics

DESCRIPTION
   Computes the mean, variance, skewness, kurtosis CCVariable<double,Vector>


WARNING

****************************************/
  class statistics : public Module {
  public:
    statistics(ProblemSpecP    & prob_spec,
               MaterialManagerP& materialManager,
               Output          * dataArchiver,
               DataArchive     * dataArchive);

    statistics();

    virtual ~statistics();

    virtual void problemSetup();

    virtual void outputProblemSpec( ProblemSpecP& ps) {};

    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level);

    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level);

    void scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                     const LevelP& level) {};

    std::string getName(){ return "statistics"; };
  private:

    //__________________________________
    //  container to hold each variable to compute stats
    struct Qstats{
//      std::string  name;
      int matl;
      VarLabel* Q_Label;
      VarLabel* Qsum_Label;
      VarLabel* Qmean_Label;

      VarLabel* Qsum2_Label;
      VarLabel* Qvariance_Label;

      VarLabel* Qsum3_Label;
      VarLabel* Qskewness_Label;

      VarLabel* Qsum4_Label;
      VarLabel* Qkurtosis_Label;

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
        //std::cout << "  setStart: " << isSet << " timestep: " << timestep << " " << name << std::endl;
      }

      void print(){
        const std::string name = Q_Label->getName();
        std::cout << name << " matl: " << matl << " subtype: " << subtype->getName() << " startTimestep: " << timestep <<"\n";
      };

    };

    //__________________________________
    //
    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);


    void doAnalysis(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);

    template <class T>
    void computeStatsWrapper( DataWarehouse* old_dw,
                              DataWarehouse* new_dw,
                              const PatchSubset* patches,
                              const Patch*   patch,
                              Qstats& Q);
    template <class T>
    void computeStats( DataWarehouse* old_dw,
                       DataWarehouse* new_dw,
                       const Patch*   patch,
                       Qstats& Q);

    template <class T>
    void allocateAndZeroSums( DataWarehouse * new_dw,
                              const Patch   * patch,
                              const Qstats  & Q);

    template <class T>
    void allocateAndZeroStats( DataWarehouse * new_dw,
                               const Patch   * patch,
                               const Qstats  & Q);

    //__________________________________
    // global constants
    const VarLabel* m_simulationTimeLabel;
    const VarLabel* m_timeStepLabel;

    double    d_startTime    = 0;
    double    d_stopTime     = DBL_MAX;
    IntVector d_monitorCell  = IntVector(-9,-9,-9);         // Cell to output

    bool d_doHigherOrderStats = false;
    std::vector< Qstats >  d_Qstats;

    ProblemSpecP          d_prob_spec;
    MaterialSet         * d_matlSet = nullptr;
  };
}
}
#endif
