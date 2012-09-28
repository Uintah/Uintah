/*
 
 The MIT License
 
 Copyright (c) 2012-2012 Center for the Simulation of Accidental Fires and 
 Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
 University of Utah.
 
 License for the specific language governing rights and limitations under
 Permission is hereby granted, free of charge, to any person obtaining a 
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation 
 the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 and/or sell copies of the Software, and to permit persons to whom the 
 Software is furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included 
 in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 DEALINGS IN THE SOFTWARE.
 
 */



#ifndef Packages_Uintah_CCA_Components_Models_SolidReactionModel_h
#define Packages_Uintah_CCA_Components_Models_SolidReactionModel_h

#include <CCA/Components/Models/SolidReactionModel/RateConstant.h>
#include <CCA/Components/Models/SolidReactionModel/RateModel.h>
#include <CCA/Ports/ModelInterface.h>

namespace Uintah {
    class ICELabel;
    
    /**************************************
     
     CLASS
     SolidReactionModel
     
     A generalized Rate Model class that very much borrows
     from the factory model idiom for rate expression composition.
     
     GENERAL INFORMATION
     
     SolidReactionModel.h
     
     Joseph R. Peterson
     Department of Chemistry
     University of Utah
     
     Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
     
     Copyright (C) 2012 SCI Group
     
     KEYWORDS
     SolidReactionModel
     
     DESCRIPTION
     Long description...
     
     WARNING
     
     ****************************************/
    
    class SolidReactionModel : public ModelInterface {
    public:
        SolidReactionModel(const ProcessorGroup* d_myworld, ProblemSpecP& params,
                           const ProblemSpecP& prob_spec);
        virtual ~SolidReactionModel();
        
        virtual void outputProblemSpec(ProblemSpecP& ps);
        
        virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
                                  ModelSetup* setup);
        
        virtual void activateModel(GridP& grid, SimulationStateP& sharedState,
                                   ModelSetup* setup);
        
        virtual void scheduleInitialize(SchedulerP&,
                                        const LevelP& level,
                                        const ModelInfo*);
        
        virtual void restartInitialize() {}
        
        virtual void scheduleComputeStableTimestep(SchedulerP& sched,
                                                   const LevelP& level,
                                                   const ModelInfo*);
        
        virtual void scheduleComputeModelSources(SchedulerP&,
                                                 const LevelP& level,
                                                 const ModelInfo*);
        
        virtual void scheduleModifyThermoTransportProperties(SchedulerP&,
                                                             const LevelP&,
                                                             const MaterialSet*);
        
        virtual void computeSpecificHeat(CCVariable<double>&,
                                         const Patch* patch,
                                         DataWarehouse* new_dw,
                                         const int indx);
        
        virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                           SchedulerP& sched);                                  
        
        virtual void scheduleCheckNeedAddMaterial(SchedulerP&,
                                                  const LevelP& level,
                                                  const ModelInfo*);
        
        virtual void scheduleTestConservation(SchedulerP&,
                                              const PatchSet* patches,
                                              const ModelInfo* mi);    
        
    private:

        void computeModelSources(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw,
                                 const ModelInfo*);

        // Functions
        SolidReactionModel(const SolidReactionModel&);
        SolidReactionModel& operator=(const SolidReactionModel&);
        
        // Innards
        RateConstant *rateConstant;  // k(T)
        RateModel    *rateModel;     // f(a)
        const Material* reactant;
        const Material* product;
        string fromMaterial;
        string toMaterial;
        double d_E0;                 // Enthalpy change for reaction in J/kg
       
        bool d_active;
        ICELabel *Ilb;               // Used to get handles on temperature, pressure, etc.
        MaterialSet *mymatls;        // All the materials referenced by this model

        // Variables used for tracking the Reaction
        const VarLabel* reactedFractionLabel;   // Fraction of reactant in cell
        const VarLabel* delFLabel;              // Change of fraction of reactant during timestep
        const VarLabel* totalMassBurnedLabel;  
        const VarLabel* totalHeatReleasedLabel;
 

        // flags for the conservation test
        struct saveConservedVars{
          bool onOff;
          bool mass;
          bool energy;
        };
        saveConservedVars* d_saveConservedVars;


        // Some Uintah Necessities
        ProblemSpecP d_params;
        ProblemSpecP d_prob_spec;
        SimulationStateP d_sharedState;



        #define d_SMALL_NUM 1e-100
        #define d_TINY_RHO 1e-12
    };
}

#endif
