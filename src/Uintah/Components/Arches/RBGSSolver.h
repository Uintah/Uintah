//
// $Id$
//

/**************************************
CLASS
   RBGSSolver
   
   Class RBGSSolver is a point red-black Gauss-Seidel
   solver

GENERAL INFORMATION
   RBGSSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class RBGSSolver is a point red-black Gauss-Seidel
   solver




WARNING
none
****************************************/

#ifndef included_RBGSSolver
#define included_RBGSSolver

#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Parallel/ProcessorContext.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
namespace Components {
class LinearSolver;
using namespace Uintah::Grid;
  using namespace Uintah::Interface;
  using namespace SCICore::Containers;
  using namespace Uintah::Parallel;

class RBGSSolver:
public LinearSolver
{
public:
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Construct an instance of a RBGSSolver.
  //
  // PRECONDITIONS
  //
  //
  // POSTCONDITIONS
  //
  // Default constructor.
 
   RBGSSolver();


  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual Destructor
   virtual ~RBGSSolver();

   virtual void problemSetup(const ProblemSpecP& params);
   // GROUP:  Methods
   ////////////////////////////////////////////////////////////////////////
   // Underrelaxation
   virtual void sched_underrelax(const LevelP& level,
				 SchedulerP& sched,
				 const DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw);
   // assigns schedules for linear solve
   virtual void sched_pressureSolve(const LevelP& level,
		      SchedulerP& sched,
		      const DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw);
   virtual void sched_velSolve(const int index, const LevelP& level,
		      SchedulerP& sched,
		      const DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw);
   virtual void sched_scalarSolve(const int index, const LevelP& level,
		      SchedulerP& sched,
		      const DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw);
 private:
   void press_underrelax(const ProcessorContext* pc,
			 const Region* region,
			 const DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw);
   void press_lisolve(const ProcessorContext* pc,
		      const Region* region,
		      const DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw);
   void press_residCalculation(const ProcessorContext* pc,
			       const Region* region,
			       const DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw);
   void vel_underrelax(const ProcessorContext* pc,
		       const Region* region,
		       const DataWarehouseP& old_dw,
		       DataWarehouseP& new_dw, int index);
   void vel_lisolve(const ProcessorContext* pc,
		    const Region* region,
		    const DataWarehouseP& old_dw,
		    DataWarehouseP& new_dw, int index);
   void vel_residCalculation(const ProcessorContext* pc,
			     const Region* region,
			     const DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw, int index);
   void scalar_underrelax(const ProcessorContext* pc,
			  const Region* region,
			  const DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw, int index);
   void scalar_lisolve(const ProcessorContext* pc,
		       const Region* region,
		       const DataWarehouseP& old_dw,
		       DataWarehouseP& new_dw, int index);
   void scalar_residCalculation(const ProcessorContext* pc,
				const Region* region,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, int index);
   int d_maxSweeps;
   double d_convgTol; // convergence tolerence
   double d_underrelax;
   double d_initResid;
   double d_residual;
};

}
}
#endif  
  
