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

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

class RadiationSolver;
class RadiationModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      RadiationModel();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for mixing model
      //
      virtual ~RadiationModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& params) = 0;
 
      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      //
      virtual void computeRadiationProps(const ProcessorGroup*,
					 const Patch* patch,
					 CellInformation* cellinfo,
					ArchesVariables* vars,
					ArchesConstVariables* constvars) = 0;


      /////////////////////////////////////////////////////////////////////////
      //
      virtual void boundarycondition(const ProcessorGroup*,
					 const Patch* patch,
					 CellInformation* cellinfo,
					ArchesVariables* vars,
					ArchesConstVariables* constvars)  = 0;

      /////////////////////////////////////////////////////////////////////////
      //
      virtual void intensitysolve(const ProcessorGroup*,
					 const Patch* patch,
					 CellInformation* cellinfo,
					ArchesVariables* vars,
					ArchesConstVariables* constvars)  = 0;
  RadiationSolver* d_linearSolver;
 protected:
      void computeOpticalLength();
      double d_opl; // optical length
 private:

}; // end class RadiationModel

} // end namespace Uintah

#endif




