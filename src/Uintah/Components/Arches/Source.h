//
// $Id$
//

#ifndef Uintah_Components_Arches_Source_h
#define Uintah_Components_Arches_Source_h

/**************************************
CLASS
   Source
   
   Class Source computes source terms for 
   N-S equations.  

GENERAL INFORMATION
   Source.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class Source computes source terms for 
   N-S equations.  



WARNING
none
****************************************/

#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
class ProcessorGroup;
namespace ArchesSpace {

class PhysicalConstants;
class TurbulenceModel;
using namespace SCICore::Containers;

class Source {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Construct an instance of a Source.
      //
      // PRECONDITIONS
      //
      // POSTCONDITIONS
      //
      Source();

      ////////////////////////////////////////////////////////////////////////
      //
      // Construct an instance of a Source.
      //
      // PRECONDITIONS
      //
      // POSTCONDITIONS
      //
      Source(TurbulenceModel* turb_model, PhysicalConstants* phys_const);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      ~Source();

      // GROUP:  Action Methods
      ////////////////////////////////////////////////////////////////////////
      //
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      //
      void calculatePressureSource(const ProcessorGroup* pc,
				   const Patch* patch,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw,
				   double delta_t); 

      ////////////////////////////////////////////////////////////////////////
      //
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      //
      void calculateVelocitySource(const ProcessorGroup* pc,
				   const Patch* patch,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw,
				   double delta_t, 
				   int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      //
      void calculateScalarSource(const ProcessorGroup* pc,
				 const Patch* patch,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw,
				 double delta_t, 
				 int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      //
      void modifyVelMassSource(const ProcessorGroup* pc,
			       const Patch* patch,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw,
			       double delta_t, 
			       int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      //
      void modifyScalarMassSource(const ProcessorGroup* pc,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw,
				  double delta_t, 
				  int index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      //
      void addPressureSource(const ProcessorGroup* pc,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw,
			     int index);
  
private:

      TurbulenceModel* d_turbModel;
      PhysicalConstants* d_physicalConsts;

      // const VarLabel*
      const VarLabel* d_densityLabel ;
      const VarLabel* d_viscosityLabel ;
      const VarLabel* d_pressureLabel ;
      const VarLabel* d_presLinearSrcLabel ;
      const VarLabel* d_presNonlinearSrcLabel ;
      const VarLabel* d_uVelocityLabel ;
      const VarLabel* d_vVelocityLabel ;
      const VarLabel* d_wVelocityLabel ;
      const VarLabel* d_velLinearSrcLabel ;
      const VarLabel* d_velNonlinearSrcLabel ;
      const VarLabel* d_uVelCoeffLabel ;
      const VarLabel* d_uNonlinearSrcLabel ;
      const VarLabel* d_vVelCoeffLabel ;
      const VarLabel* d_vNonlinearSrcLabel ;
      const VarLabel* d_wVelCoeffLabel ;
      const VarLabel* d_wNonlinearSrcLabel ;
      const VarLabel* d_scalarLabel ;
      const VarLabel* d_scalarLinearSrcLabel ;
      const VarLabel* d_scalarNonlinearSrcLabel ;

}; // end Class Source

}  // End namespace ArchesSpace
}  // End namespace Uintah
#endif  
  
//
// $Log$
// Revision 1.12  2000/06/17 07:06:27  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.11  2000/06/13 06:02:32  bbanerje
// Added some more StencilMatrices and vector<CCVariable> types.
//
// Revision 1.10  2000/06/12 21:30:00  bbanerje
// Added first Fortran routines, added Stencil Matrix where needed,
// removed unnecessary CCVariables (e.g., sources etc.)
//
// Revision 1.9  2000/06/07 06:13:57  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.8  2000/06/04 22:40:16  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
