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

#include <Uintah/Components/Arches/StencilMatrix.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/DataWarehouseP.h>
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
				  int index);

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
				int Index);

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
      StencilMatrix<CCVariable<double> >* d_press_stencil_matrix;
      // stores coefficients for all the velocity components
      // coefficients should be saved on staggered grid
      StencilMatrix<CCVariable<double> >* d_mom_stencil_matrix;
      // coefficients for all the scalar components
      StencilMatrix<CCVariable<double> >* d_scalar_stencil_matrix;

      // const VarLabel*
      const VarLabel* d_uVelocityLabel;
      const VarLabel* d_vVelocityLabel;
      const VarLabel* d_wVelocityLabel;
      const VarLabel* d_scalarLabel;
      const VarLabel* d_pressureLabel;
      const VarLabel* d_densityLabel;
      const VarLabel* d_viscosityLabel;
      const VarLabel* d_uVelCoefLabel;
      const VarLabel* d_vVelCoefLabel;
      const VarLabel* d_wVelCoefLabel;
      const VarLabel* d_uVelConvCoefLabel;
      const VarLabel* d_vVelConvCoefLabel;
      const VarLabel* d_wVelConvCoefLabel;
      const VarLabel* d_presCoefLabel;
      const VarLabel* d_scalCoefLabel;
      const VarLabel* d_uVelLinSrcLabel;
      const VarLabel* d_vVelLinSrcLabel;
      const VarLabel* d_wVelLinSrcLabel;
      const VarLabel* d_presLinSrcLabel;
      const VarLabel* d_scalLinSrcLabel;

}; // end class Discretization
} // end namespace ArchesSpace
} // end namespace Uintah

#endif  
  
//
// $Log$
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
