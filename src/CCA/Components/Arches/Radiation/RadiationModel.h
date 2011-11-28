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


//----- RadiationModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_RadiationModel_h
#define Uintah_Component_Arches_RadiationModel_h

/***************************************************************************
CLASS
    RadiationModel
       Sets up the RadiationModel ????
       
GENERAL INFORMATION
    RadiationModel.h - Declaration of RadiationModel class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    
    Creation Date : 05-30-2000

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    
DESCRIPTION

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <CCA/Components/Arches/Arches.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>


namespace Uintah {

class RadiationSolver;
class RadiationModel {

public:

      RadiationModel();

      virtual ~RadiationModel();


      virtual void problemSetup(ProblemSpecP& params) = 0;
 

      virtual void computeRadiationProps(const ProcessorGroup*,
                                         const Patch* patch,
                                         CellInformation* cellinfo,
                                         ArchesVariables* vars,
                                         ArchesConstVariables* constvars) = 0;

      virtual void boundarycondition(const ProcessorGroup*,
                                     const Patch* patch,
                                     CellInformation* cellinfo,
                                     ArchesVariables* vars,
                                     ArchesConstVariables* constvars)  = 0;

      virtual void intensitysolve(const ProcessorGroup*,
                                  const Patch* patch,
                                  CellInformation* cellinfo,
                                  ArchesVariables* vars,
                                  ArchesConstVariables* constvars, int wall_type )  = 0;

      //______________________________________________________________________
      //
      virtual void sched_computeSource( const LevelP& level, 
                                        SchedulerP& sched, 
                                        const MaterialSet* matls,
                                        const TimeIntegratorLabel* timelabels,
                                        const bool isFirstIntegrationStep ) = 0;
                                
      virtual void computeSource( const ProcessorGroup* pc, 
                                  const PatchSubset* patches,             
                                  const MaterialSubset* matls,            
                                  DataWarehouse* old_dw,                  
                                  DataWarehouse* new_dw,
                                  const TimeIntegratorLabel* timelabels,          
                                  bool isFirstIntegrationStep ) = 0;
                                  
  RadiationSolver* d_linearSolver;
 protected:
      void computeOpticalLength();
      double d_opl; // optical length
 private:

}; // end class RadiationModel

} // end namespace Uintah

#endif




