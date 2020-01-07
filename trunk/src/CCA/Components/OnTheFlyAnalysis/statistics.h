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


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_statistics_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_statistics_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <vector>

namespace Uintah {


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
   This computes turbulence related statistical quantities


WARNING

****************************************/
  class statistics : public AnalysisModule {
  public:
    statistics(const ProcessorGroup* myworld,
               const MaterialManagerP materialManager,
               const ProblemSpecP& module_spec);

    statistics();

    virtual ~statistics();

    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);
                              
    virtual void outputProblemSpec( ProblemSpecP& ps);

    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level);

    virtual void scheduleRestartInitialize(SchedulerP& sched,
                                           const LevelP& level);

    virtual void restartInitialize();

    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level);

    virtual void scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                             const LevelP& level) {};

  private:
    enum ORDER {lowOrder, highOrder};

    //__________________________________
    //  container to hold
    struct Qstats{
//      std::string  name;
      bool computeRstess;
      int matl;
      VarLabel* Q_Label;
      VarLabel* Qsum_Label;
      VarLabel* Qmean_Label;

      VarLabel* Qsum2_Label;
      VarLabel* Qmean2_Label;
      VarLabel* Qvariance_Label;

      VarLabel* Qsum3_Label;
      VarLabel* Qmean3_Label;
      VarLabel* Qskewness_Label;

      VarLabel* Qsum4_Label;
      VarLabel* Qmean4_Label;
      VarLabel* Qkurtosis_Label;

      std::map<ORDER,bool> isInitialized;

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
    // For Reynolds Shear Stress computations
    bool d_isReynoldsStressInitialized; // have the sum label been initialized for the RS terms
    bool d_computeReynoldsStress;       // on/off switch
    int  d_RS_matl;                     // material index used for Reynolds Shear Stress variables
    VarLabel* d_velPrime_Label;         // u'v', v'w', w'u'
    VarLabel* d_velSum_Label;           // sum(u'v'), sum(v'w'), sum(w'u')              over timesteps
    VarLabel* d_velMean_Label;          // sum(u'v')/N, sum(v'w')/N, sum(w'u')/N        over timesteps where N = number of timesteps

    inline Vector Multiply(Vector a, Vector b){
      return Vector(a.x()*b.y(), a.y()*b.z(), a.z()*b.x() );
    }

    //__________________________________
    //
    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);

    void restartInitialize(const ProcessorGroup*,
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

    void computeReynoldsStressWrapper( DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       const PatchSubset* patches,
                                       const Patch*    patch,
                                       Qstats& Q);

    void computeReynoldsStress( DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const Patch*    patch,
                                Qstats& Q);

    template <class T>
    void allocateAndZero( DataWarehouse* new_dw,
                          const VarLabel* label,
                          const int       matl,
                          const Patch*    patch );
    template <class T>
    void allocateAndZeroSums( DataWarehouse* new_dw,
                              const Patch*   patch,
                              Qstats& Q);

    template <class T>
    void allocateAndZeroStats( DataWarehouse* new_dw,
                               const Patch*   patch,
                               const Qstats& Q);

    void carryForwardSums( DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           const PatchSubset* patches,
                           const Qstats& Q );

    //__________________________________
    // global constants
//    int       d_startTimeTimestep;   // timestep when stats are turn on.
    IntVector d_monitorCell;         // Cell to output

    bool d_doHigherOrderStats;
    std::vector< Qstats >  d_Qstats;

    const Material       * d_matl;
    MaterialSet          * d_matlSet;
    const MaterialSubset * d_matSubSet;

    bool required;
  };
}

#endif
