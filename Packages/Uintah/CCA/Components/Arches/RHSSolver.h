
#ifndef Uintah_Components_Arches_RHSSolver_h
#define Uintah_Components_Arches_RHSSolver_h

/**************************************
CLASS
   RHSSolver
   
   Class RHSSolver is a "right hand side" solver
   doing a final "solve" for new (timestep N+1) values 

GENERAL INFORMATION
   RHSSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION

WARNING
   none

****************************************/

#include <Packages/Uintah/CCA/Components/Arches/LinearSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>

#include <Core/Containers/Array1.h>

namespace Uintah {

using namespace SCIRun;

class RHSSolver{

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a RHSSolver.
      RHSSolver();

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~RHSSolver();


      ////////////////////////////////////////////////////////////////////////
      // Velocity Solve
      void calculateHatVelocity(const ProcessorGroup* ,
			      const Patch* patch,
			      int index, double delta_t,
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars);

      void calculateVelocity(const ProcessorGroup* ,
				 const Patch* patch,
				 double delta_t,
				 int index,
				 CellInformation* cellinfo,
				 ArchesVariables* vars,
				 ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Scalar Solve
      void scalarLisolve(const ProcessorGroup* pc,
			 const Patch* patch,
			 double delta_t,
			 ArchesVariables* vars,
			 ArchesConstVariables* constvars,
			 CellInformation* cellinfo);

      ////////////////////////////////////////////////////////////////////////
      // Scalar Solve
      void enthalpyLisolve(const ProcessorGroup* pc,
			 const Patch* patch,
			 double delta_t,
			 ArchesVariables* vars,
			 ArchesConstVariables* constvars,
			 CellInformation* cellinfo);

protected:

private:


}; // End class RHSSolver.h

} // End namespace Uintah

#endif  
  
