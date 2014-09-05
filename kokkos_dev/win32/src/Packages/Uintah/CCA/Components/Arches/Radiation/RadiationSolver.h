
#ifndef Uintah_Components_Arches_RadiationSolver_h
#define Uintah_Components_Arches_RadiationSolver_h

#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>

#include <Core/Containers/Array1.h>

namespace Uintah {

class ProcessorGroup;
class LoadBalancer;
class ArchesVariables;
class ArchesConstVariables;
class ArchesLabel;
class CellInformation;
// class StencilMatrix;

using namespace SCIRun;

/**************************************
CLASS
   RadiationSolver
   
   Class RadiationSolver is an abstract base class
   that solves the linearized PDE.

GENERAL INFORMATION
   RadiationSolver.h - declaration of the class
   
   Author: Gautham Krishnamoorthy (gautham@crsim.utah.edu)
   
   Creation Date:   July 2, 2004
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class RadiationSolver is an abstract base class
   that solves the linearized PDE.



WARNING
none
****************************************/

class RadiationSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a LinearSolver.
      // PRECONDITIONS
      // POSTCONDITIONS
      RadiationSolver();


      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~RadiationSolver();


      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Setup the problem (etc.)
      virtual void problemSetup(const ProblemSpecP& params) = 0;

//      inline double getInitNorm() { return init_norm; }

      ////////////////////////////////////////////////////////////////////////

   virtual void matrixCreate(const PatchSet* allpatches,
			     const PatchSubset* mypatches) = 0;

   virtual void setMatrix(const ProcessorGroup* pc,
			    const Patch* patch,
			    ArchesVariables* vars,
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
   virtual void copyRadSoln(const Patch* patch, ArchesVariables* vars) = 0;
   virtual void destroyMatrix() = 0;
//   double init_norm;

protected:

private:

}; // End class RadiationSolver

} // End namespace Uintah

#endif  
