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
				   int index,
				   int eqnType);

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
      const VarLabel* d_cellInfoLabel ;

      // inputs for CalculateVelocitySource 
      const VarLabel* d_uVelocitySIVBCLabel ;
      const VarLabel* d_vVelocitySIVBCLabel ;
      const VarLabel* d_wVelocitySIVBCLabel ;
      const VarLabel* d_densitySIVBCLabel ;
      const VarLabel* d_viscosityCTSLabel ;
      const VarLabel* d_uVelocityCPBCLabel ;
      const VarLabel* d_vVelocityCPBCLabel ;
      const VarLabel* d_wVelocityCPBCLabel ;

      // outputs for CalculateVelocitySource
      const VarLabel* d_uVelLinSrcPBLMLabel ;
      const VarLabel* d_uVelNonLinSrcPBLMLabel ;
      const VarLabel* d_vVelLinSrcPBLMLabel ;
      const VarLabel* d_vVelNonLinSrcPBLMLabel ;
      const VarLabel* d_wVelLinSrcPBLMLabel ;
      const VarLabel* d_wVelNonLinSrcPBLMLabel ;
      const VarLabel* d_uVelLinSrcMBLMLabel ;
      const VarLabel* d_uVelNonLinSrcMBLMLabel ;
      const VarLabel* d_vVelLinSrcMBLMLabel ;
      const VarLabel* d_vVelNonLinSrcMBLMLabel ;
      const VarLabel* d_wVelLinSrcMBLMLabel ;
      const VarLabel* d_wVelNonLinSrcMBLMLabel ;

      // inputs/outputs for CalculatePressureSource 
      const VarLabel* d_uVelCoefPBLMLabel ;
      const VarLabel* d_vVelCoefPBLMLabel ;
      const VarLabel* d_wVelCoefPBLMLabel ;
      const VarLabel* d_pressureINLabel ;
      const VarLabel* d_presLinSrcPBLMLabel ;
      const VarLabel* d_presNonLinSrcPBLMLabel ;

      // inputs/outputs for CalculateScalarSource 
      const VarLabel* d_uVelocityMSLabel ;
      const VarLabel* d_vVelocityMSLabel ;
      const VarLabel* d_wVelocityMSLabel ;
      const VarLabel* d_scalarSPLabel ;
      const VarLabel* d_scalLinSrcSBLMLabel ;
      const VarLabel* d_scalNonLinSrcSBLMLabel ;

}; // end Class Source

}  // End namespace ArchesSpace
}  // End namespace Uintah
#endif  
  
//
// $Log$
// Revision 1.15  2000/07/02 05:47:31  bbanerje
// Uncommented all PerPatch and CellInformation stuff.
// Updated array sizes in inlbcs.F
//
// Revision 1.14  2000/06/21 07:51:02  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.13  2000/06/18 01:20:17  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
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
