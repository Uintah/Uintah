//
// $Id$
//

#ifndef Uintah_Components_Arches_Discretization_h
#define Uintah_Components_Arches_Discretization_h

/**************************************
CLASS
   Discretization
   
   Class Discretization is a class
   that computes stencil weights for linearized 
   N-S equations.  

GENERAL INFORMATION
   Discretization.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class Discretization is an abstract base class
   that computes stencil weights for linearized 
   N-S equations.  


WARNING
none
****************************************/

//#include <Uintah/Components/Arches/StencilMatrix.h>
//#include <Uintah/Grid/CCVariable.h>
//#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/VarLabel.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
   class ProcessorGroup;
namespace ArchesSpace {

//class StencilMatrix;
using namespace SCICore::Containers;

class Discretization {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Construct an instance of a Discretization.
      //
      // PRECONDITIONS
      //
      // POSTCONDITIONS
      //
      // Default constructor.
      //
      Discretization();

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Virtual Destructor
      //
      virtual ~Discretization();

      // GROUP:  Action Methods
      ////////////////////////////////////////////////////////////////////////
      //
      // Set stencil weights. (Velocity)
      // It uses second order hybrid differencing for computing
      // coefficients
      //
      void calculateVelocityCoeff(const ProcessorGroup*,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw,
				  double delta_t,
				  int index,
				  int eqnType);

      ////////////////////////////////////////////////////////////////////////
      //
      // Set stencil weights. (Pressure)
      // It uses second order hybrid differencing for computing
      // coefficients
      //
      void calculatePressureCoeff(const ProcessorGroup*,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw,
				  double delta_t); 

      ////////////////////////////////////////////////////////////////////////
      //
      // Set stencil weights. (Scalars)
      // It uses second order hybrid differencing for computing
      // coefficients
      //
      void calculateScalarCoeff(const ProcessorGroup*,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				double delta_t,
				int Index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Documentation here
      //
      void calculateVelDiagonal(const ProcessorGroup*,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				int index,
				int eqnType);

      ////////////////////////////////////////////////////////////////////////
      //
      // Documentation here
      //
      void calculatePressDiagonal(const ProcessorGroup*,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Documentation here
      //
      void calculateScalarDiagonal(const ProcessorGroup*,
				   const Patch* patch,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw,
				   int Index);
protected:

private:
   
      // Stencil weights.
      // Array of size NDIM and of depth determined by stencil coefficients
      //StencilMatrix<CCVariable<double> >* d_press_stencil_matrix;
      // stores coefficients for all the velocity components
      // coefficients should be saved on staggered grid
      //StencilMatrix<FCVariable<double> >* d_mom_stencil_matrix;
      // coefficients for all the scalar components
      //StencilMatrix<CCVariable<double> >* d_scalar_stencil_matrix;

      // const VarLabel*
      const VarLabel* d_cellInfoLabel;

      // inputs
      const VarLabel* d_uVelocitySIVBCLabel;
      const VarLabel* d_vVelocitySIVBCLabel;
      const VarLabel* d_wVelocitySIVBCLabel;
      const VarLabel* d_densityCPLabel;
      const VarLabel* d_viscosityCTSLabel;
      const VarLabel* d_uVelocityCPBCLabel;
      const VarLabel* d_vVelocityCPBCLabel;
      const VarLabel* d_wVelocityCPBCLabel;

      // used/output by calculateVelocityCoeff
      const VarLabel* d_DUPBLMLabel;
      const VarLabel* d_uVelCoefPBLMLabel;
      const VarLabel* d_vVelCoefPBLMLabel;
      const VarLabel* d_wVelCoefPBLMLabel;
      const VarLabel* d_uVelConvCoefPBLMLabel;
      const VarLabel* d_vVelConvCoefPBLMLabel;
      const VarLabel* d_wVelConvCoefPBLMLabel;
      const VarLabel* d_uVelCoefMBLMLabel;
      const VarLabel* d_vVelCoefMBLMLabel;
      const VarLabel* d_wVelCoefMBLMLabel;
      const VarLabel* d_uVelConvCoefMBLMLabel;
      const VarLabel* d_vVelConvCoefMBLMLabel;
      const VarLabel* d_wVelConvCoefMBLMLabel;

      // input/output for calculatePressureCoeff
      const VarLabel* d_pressureSPBCLabel;
      const VarLabel* d_presCoefPBLMLabel;
      const VarLabel* d_presLinSrcPBLMLabel;

      // Input/output for calculateVelDiagonal
      const VarLabel* d_uVelLinSrcPBLMLabel;
      const VarLabel* d_vVelLinSrcPBLMLabel;
      const VarLabel* d_wVelLinSrcPBLMLabel;
      const VarLabel* d_uVelLinSrcMBLMLabel;
      const VarLabel* d_vVelLinSrcMBLMLabel;
      const VarLabel* d_wVelLinSrcMBLMLabel;

      // Input/output for calculateScalarCoeff
      const VarLabel* d_scalarSPLabel;
      const VarLabel* d_uVelocityMSLabel;
      const VarLabel* d_vVelocityMSLabel;
      const VarLabel* d_wVelocityMSLabel;
      const VarLabel* d_scalCoefSBLMLabel;
      const VarLabel* d_scalConvCoefSBLMLabel;

      // Input/output for calculateScalarDiagonal
      const VarLabel* d_scalLinSrcSBLMLabel;

}; // end class Discretization
} // end namespace ArchesSpace
} // end namespace Uintah

#endif  
  
//
// $Log$
// Revision 1.25  2000/07/14 05:23:50  bbanerje
// Added scalcoef.F and updated related stuff in C++. scalcoef ==> coefs.f
// in Kumar's code.
//
// Revision 1.24  2000/07/08 23:42:54  bbanerje
// Moved all enums to Arches.h and made corresponding changes.
//
// Revision 1.23  2000/07/08 08:03:34  bbanerje
// Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
// made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
// to Arches::AE .. doesn't like enums in templates apparently.
//
// Revision 1.22  2000/07/03 05:30:14  bbanerje
// Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
// Revision 1.21  2000/07/02 05:47:30  bbanerje
// Uncommented all PerPatch and CellInformation stuff.
// Updated array sizes in inlbcs.F
//
// Revision 1.20  2000/06/29 21:48:59  bbanerje
// Changed FC Vars to SFCX,Y,ZVars and added correct getIndex() to get domainhi/lo
// and index hi/lo
//
// Revision 1.19  2000/06/22 23:06:34  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.18  2000/06/21 07:51:00  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.17  2000/06/18 01:20:15  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.16  2000/06/17 07:06:23  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.15  2000/06/13 06:02:31  bbanerje
// Added some more StencilMatrices and vector<CCVariable> types.
//
// Revision 1.14  2000/06/12 21:29:59  bbanerje
// Added first Fortran routines, added Stencil Matrix where needed,
// removed unnecessary CCVariables (e.g., sources etc.)
//
// Revision 1.13  2000/06/07 06:13:54  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.12  2000/06/04 22:40:13  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
