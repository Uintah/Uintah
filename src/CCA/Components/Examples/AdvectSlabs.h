/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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



#ifndef Packages_Uintah_CCA_Components_Examples_AdvectSlabs_h
#define Packages_Uintah_CCA_Components_Examples_AdvectSlabs_h

#include <CCA/Components/ICE/Advection/Advector.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>

#include <CCA/Components/Examples/uintahshare.h>
namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;

/**************************************

CLASS
   AdvectSlabs
   
   AdvectSlabs simulation

GENERAL INFORMATION

   AdvectSlabs.h

   Alan Humphrey
   Department of Computer Science
   University of Utah

   Joseph R. Peterson
   Department of Chemistry
   University of Utah
 
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2011 SCI Group

KEYWORDS
   AdvectSlabs

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class UINTAHSHARE AdvectSlabs : public UintahParallelComponent, public SimulationInterface {
  public:
    AdvectSlabs(const ProcessorGroup* myworld);
    virtual ~AdvectSlabs();

    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, 
                              GridP& grid, SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, 
				      SchedulerP&);
  private:
    void initialize(const ProcessorGroup*,
		    const PatchSubset* patches, const MaterialSubset* matls,
		    DataWarehouse* old_dw, DataWarehouse* new_dw);
    void computeStableTimestep(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw);
    void timeAdvance(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw, DataWarehouse* new_dw);

    void iterate(const ProcessorGroup*,
		 const PatchSubset* patches,
		 const MaterialSubset* matls,
		 DataWarehouse* old_dw, DataWarehouse* new_dw);

    const VarLabel* mass_label;
    const VarLabel* massAdvected_label;
      
    SimulationStateP sharedState_;
    double delt_;
    double maxresidual_;
    SimpleMaterial* mymat_;

    // Our advector variables  
    CCVariable<fflux> d_OFS;
    int OF_slab[6];          // outflux slab
    int IF_slab[6];          // influx flab
    IntVector S_ac[6];       // slab adj. cell
    enum FACE {TOP, BOTTOM, RIGHT, LEFT, FRONT, BACK};
    const static int NGC =1;  // number of ghostCells
    
    AdvectSlabs(const AdvectSlabs&);
    AdvectSlabs& operator=(const AdvectSlabs&);
	 
  };
}

#endif
