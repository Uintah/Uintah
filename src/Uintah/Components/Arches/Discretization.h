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

#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Parallel/ProcessorContext.h>
#include <Uintah/Grid/VarLabel.h>

#include <SCICore/Containers/Array1.h>

namespace Uintah {
namespace ArchesSpace {

class StencilMatrix;
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
      void calculateVelocityCoeff(const ProcessorContext*,
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
      void calculatePressureCoeff(const ProcessorContext*,
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
      void calculateScalarCoeff(const ProcessorContext*,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				double delta_t,
				int Index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Documentation here
      //
      void calculateVelDiagonal(const ProcessorContext*,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				int Index);

      ////////////////////////////////////////////////////////////////////////
      //
      // Documentation here
      //
      void calculatePressDiagonal(const ProcessorContext*,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw);

      ////////////////////////////////////////////////////////////////////////
      //
      // Documentation here
      //
      void calculateScalarDiagonal(const ProcessorContext*,
				   const Patch* patch,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw,
				   int Index);
protected:

private:
   
      // Stencil weights.
      // Array of size NDIM and of depth determined by stencil coefficients
      StencilMatrix* d_press_stencil_matrix;
      // stores coefficients for all the velocity components
      // coefficients should be saved on staggered grid
      StencilMatrix* d_mom_stencil_matrix;
      // coefficients for all the scalar components
      StencilMatrix* d_scalar_stencil_matrix;

      // const VarLabel*
      const VarLabel* d_velocityLabel;
      const VarLabel* d_scalarLabel;
      const VarLabel* d_pressureLabel;
      const VarLabel* d_densityLabel;
      const VarLabel* d_viscosityLabel;

}; // end class Discretization
} // end namespace ArchesSpace
} // end namespace Uintah

#endif  
  
//
// $Log$
// Revision 1.12  2000/06/04 22:40:13  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
