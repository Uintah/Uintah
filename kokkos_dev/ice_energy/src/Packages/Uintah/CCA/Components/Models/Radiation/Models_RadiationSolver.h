
#ifndef Uintah_Components_Models_RadiationSolver_h
#define Uintah_Components_Models_RadiationSolver_h

#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationVariables.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationConstVariables.h>

#include <Core/Containers/Array1.h>

namespace Uintah {

class ProcessorGroup;
class LoadBalancer;
class RadiationVariables;
class RadiationConstVariables;
class ICELabel;
class Models_CellInformation;

using namespace SCIRun;

/**************************************
CLASS
   Models_RadiationSolver
   
   Class Models_RadiationSolver is an abstract base class
   that solves the linearized PDE.

GENERAL INFORMATION
   Models_RadiationSolver.h - declaration of the class
   
   Author: Seshadri Kumar (skumar@crsim.utah.edu)
   
   Creation Date: April 12, 2005
   
   C-SAFE 
   
   Copyright U of U 2005

KEYWORDS


DESCRIPTION
   Class Models_RadiationSolver is an abstract base class
   that solves the linearized PDE.  This routine is written based
   on RadiationSolver.h written by Gautham Krishnamoorthy (July 2, 2004) 
   (gautham@crsim.utah.edu) as part of Arches/Radiation.



WARNING
none
****************************************/

class Models_RadiationSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a LinearSolver.
      // PRECONDITIONS
      // POSTCONDITIONS
      Models_RadiationSolver();


      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~Models_RadiationSolver();


      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Setup the problem (etc.)
      virtual void problemSetup(const ProblemSpecP& params, bool shradiation) = 0;

//      inline double getInitNorm() { return init_norm; }

      ////////////////////////////////////////////////////////////////////////

   virtual void matrixCreate(const PatchSet* allpatches,
			     const PatchSubset* mypatches) = 0;

   virtual void setMatrix(const ProcessorGroup* pc,
			  const Patch* patch,
			  RadiationVariables* vars,
			  bool plusX, bool plusY, bool plusZ,
			  CCVariable<double>& SU,
			  CCVariable<double>& AB,
			  CCVariable<double>& AS,
			  CCVariable<double>& AW,
			  CCVariable<double>& AP,
			  CCVariable<double>& AE,
			  CCVariable<double>& AN,
			  CCVariable<double>& AT) = 0;

   virtual bool radLinearSolve() = 0;
   virtual void copyRadSoln(const Patch* patch, RadiationVariables* vars) = 0;
   virtual void destroyMatrix() = 0;
//   double init_norm;

protected:

private:

}; // End class Models_RadiationSolver

} // End namespace Uintah

#endif  
