//
// $Id : $
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

#include <Uintah/Parallel/ProcessorContext.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
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
      void calculatePressureSource(const ProcessorContext* pc,
				   const Patch* patch,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw,
				   double delta_t); 

      ////////////////////////////////////////////////////////////////////////
      //
      // Set source terms. Will need more parameters...like velocity and
      // scalars
      //
      void calculateVelocitySource(const ProcessorContext* pc,
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
      void calculateScalarSource(const ProcessorContext* pc,
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
      void modifyVelMassSource(const ProcessorContext* pc,
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
      void modifyScalarMassSource(const ProcessorContext* pc,
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
      void addPressureSource(const ProcessorContext* pc,
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
      const VarLabel* d_xuVelCoeffLabel ;
      const VarLabel* d_yuVelCoeffLabel ;
      const VarLabel* d_zuVelCoeffLabel ;
      const VarLabel* d_uNonlinearSrcLabel ;
      const VarLabel* d_xvVelCoeffLabel ;
      const VarLabel* d_yvVelCoeffLabel ;
      const VarLabel* d_zvVelCoeffLabel ;
      const VarLabel* d_vNonlinearSrcLabel ;
      const VarLabel* d_xwVelCoeffLabel ;
      const VarLabel* d_ywVelCoeffLabel ;
      const VarLabel* d_zwVelCoeffLabel ;
      const VarLabel* d_wNonlinearSrcLabel ;
      const VarLabel* d_xScalarLabel ;
      const VarLabel* d_yScalarLabel ;
      const VarLabel* d_zScalarLabel ;
      const VarLabel* d_xScalarLinearSrcLabel ;
      const VarLabel* d_yScalarLinearSrcLabel ;
      const VarLabel* d_zScalarLinearSrcLabel ;
      const VarLabel* d_xScalarNonlinearSrcLabel ;
      const VarLabel* d_yScalarNonlinearSrcLabel ;
      const VarLabel* d_zScalarNonlinearSrcLabel ;

}; // end Class Source

}  // End namespace ArchesSpace
}  // End namespace Uintah
#endif  
  
//
// $Log$
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
