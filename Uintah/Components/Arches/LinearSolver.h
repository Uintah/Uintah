//
// $Id$
//

/**************************************
CLASS
   LinearSolver
   
   Class LinearSolver is an abstract base class
   that solves the linearized PDE.

GENERAL INFORMATION
   LinearSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class LinearSolver is an abstract base class
   that solves the linearized PDE.



WARNING
none
****************************************/

#ifndef included_LinearSolver
#define included_LinearSolver

#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Parallel/ProcessorContext.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
namespace ArchesSpace {
class StencilMatrix;
  using namespace SCICore::Containers;

class LinearSolver
{
public:
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Construct an instance of a LinearSolver.
  //
  // PRECONDITIONS
  //
  //
  // POSTCONDITIONS
  //
  // Default constructor.
 
   LinearSolver();


  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual Destructor
   virtual ~LinearSolver();

   virtual void problemSetup(const ProblemSpecP& params) = 0;
   // GROUP:  Methods
   ////////////////////////////////////////////////////////////////////////
   // assigns schedules for linear solve
   virtual void sched_pressureSolve(const LevelP& level,
		      SchedulerP& sched,
		      const DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw) = 0;
   virtual void sched_velSolve(const int Index, const LevelP& level,
		      SchedulerP& sched,
		      const DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw) = 0;
   virtual void sched_scalarSolve(const int index, const LevelP& level,
		      SchedulerP& sched,
		      const DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw) = 0;
 private:
};

}
}
#endif  
  
