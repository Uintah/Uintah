//----- Models_RadiationModel.h --------------------------------------------------

#ifndef Uintah_Component_Models_RadiationModel_h
#define Uintah_Component_Models_RadiationModel_h

/***************************************************************************
CLASS
    Models_RadiationModel
       Sets up the Models_RadiationModel
       
GENERAL INFORMATION
    Models_RadiationModel.h - Declaration of Models_RadiationModel class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    Creation Date : 05-30-2000

    Modified: for Incorporation into Models Infrastructure, 
              Seshadri Kumar (skumar@crsim.utah.edu)
    
    Modification (start of) Date: April 11, 2005

    C-SAFE
    
    Copyright U of U 2005

KEYWORDS
    
DESCRIPTION

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_CellInformation.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationVariables.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationConstVariables.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

class Models_RadiationSolver;
class Models_RadiationModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      Models_RadiationModel();

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for mixing model
      //
      virtual ~Models_RadiationModel();

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
                                         Models_CellInformation* cellinfo,
                                         RadiationVariables* vars,
                                         RadiationConstVariables* constvars) = 0;


      /////////////////////////////////////////////////////////////////////////
      //
      virtual void boundaryCondition(const ProcessorGroup*,
                                     const Patch* patch,
                                     RadiationVariables* vars,
                                     RadiationConstVariables* constvars)  = 0;

      /////////////////////////////////////////////////////////////////////////
      //
      virtual void intensitysolve(const ProcessorGroup*,
                                  const Patch* patch,
                                  Models_CellInformation* cellinfo,
                                  RadiationVariables* vars,
                                  RadiationConstVariables* constvars)  = 0;

      Models_RadiationSolver* d_linearSolver;
 protected:
      void computeOpticalLength();
      double d_opl; // optical length
 private:

}; // end class Models_RadiationModel

} // end namespace Uintah

#endif




