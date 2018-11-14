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


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_meanTurbFluxes_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_meanTurbFluxes_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Components/OnTheFlyAnalysis/planeAverage.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/GridIterator.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>

#include <vector>
#include <memory>

namespace Uintah {


/*______________________________________________________________________

GENERAL INFORMATION

   meanTurbFluxes.h

   This module computes the mean turbulent fluxes on each plane in the domain
   u'u'_bar, v'v'_bar, w'w'_bar u'v'_bar, v'w'_bar u'w'_bar

   foreach Q ( T, P, scalar )
     ( u'Q'_bar(y), v'Q'_bar(y), w'Q'_bar(y) )
   end

   Todd Harman
   Department of Mechanical Engineering
   University of Utah
______________________________________________________________________*/



//______________________________________________________________________

  class meanTurbFluxes :  public AnalysisModule{
  public:

    meanTurbFluxes(const ProcessorGroup   * myworld,
                   const MaterialManagerP   materialManager,
                   const ProblemSpecP     & module_spec);

    meanTurbFluxes();

    virtual ~meanTurbFluxes();

    virtual void problemSetup(const ProblemSpecP  & prob_spec,
                              const ProblemSpecP  & restart_prob_spec,
                              GridP               & grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);

    virtual void outputProblemSpec(ProblemSpecP& ps){};

    virtual void scheduleInitialize(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleRestartInitialize(SchedulerP   & sched,
                                           const LevelP & level);

    virtual void restartInitialize();

    virtual void scheduleDoAnalysis(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleDoAnalysis_preReloc(SchedulerP   & sched,
                                             const LevelP & level) {};


  private:
    //__________________________________
    //  all variables except velocity
    struct Qvar{

      int matl;
      int level;
      VarLabel * label;                   // Q
      VarLabel * primeLabel;              // Q'
      VarLabel * turbFluxLabel;           // u'Q', v'Q', w'Q'

      MaterialSubset * matSubSet;
      TypeDescription::Type baseType;
      TypeDescription::Type subType;

      void print(){
        const std::string name = label->getName();
        std::cout << name << " matl: " << matl <<"\n";
      };

      ~Qvar()
      {
        if(matSubSet && matSubSet->removeReference()){
          delete matSubSet;
        }
        VarLabel::destroy( label );
        VarLabel::destroy( primeLabel );
        VarLabel::destroy( turbFluxLabel );
      }
    };

    //__________________________________
    //  Velocity
    struct velocityVar: public Qvar{
      VarLabel * normalTurbStrssLabel;        //u'u', v'v', w'w'
      VarLabel * shearTurbStrssLabel;     //u'v', v'w', w'u'

      ~velocityVar()
      {
        VarLabel::destroy( normalTurbStrssLabel );
        VarLabel::destroy( shearTurbStrssLabel );
      }
      std::string normalTurbStrssName  = "normalTurbStrss";
      std::string shearTurbStrssName   = "shearTurbStrss";
    };


    //______________________________________________________________________
    //
    //
    void initialize(const ProcessorGroup *,
                    const PatchSubset    * patches,
                    const MaterialSubset *,
                    DataWarehouse        *,
                    DataWarehouse        * new_dw);

    void sched_TurbFluctuations(SchedulerP   & sched,
                                const LevelP & level);

    void calc_TurbFluctuations(const ProcessorGroup  * ,
                               const PatchSubset    * patches,
                               const MaterialSubset * ,
                               DataWarehouse        * ,
                               DataWarehouse        * new_dw);

    template <class T>
    void calc_Q_prime( DataWarehouse       * new_dw,
                       const Patch         * patch,
                       std::shared_ptr<Qvar> Q );

    void sched_TurbFluxes(SchedulerP   & sched,
                          const LevelP & level);

    void calc_TurbFluxes(const ProcessorGroup * ,
                         const PatchSubset    * patches,
                         const MaterialSubset * ,
                         DataWarehouse        * ,
                         DataWarehouse        * new_dw);

    //______________________________________________________________________
    // general labels
    class meanTurbFluxesLabel {
    public:
      VarLabel* lastCompTimeLabel;
      VarLabel* fileVarsStructLabel;
      VarLabel* weightLabel = {nullptr};
    };

    meanTurbFluxesLabel* d_lb;

    //__________________________________
    // global constants always begin with "d_"
    std::vector< std::shared_ptr< Qvar > >  d_Qvars;
    std::shared_ptr< velocityVar >          d_velVar;

    double d_writeFreq;
    double d_startTime;
    double d_stopTime;

    const Material*  d_matl;
    MaterialSet*     d_matl_set;
    std::set<std::string> d_isDirCreated;
    MaterialSubset*  d_zero_matl;

    const int d_MAXLEVELS {5};               // HARDCODED

    enum orientation { XY, XZ, YZ };        // plane orientation
    orientation d_planeOrientation;


    private:
      planeAverage * d_planeAve_1;
      planeAverage * d_planeAve_2;


  };
}

#endif
