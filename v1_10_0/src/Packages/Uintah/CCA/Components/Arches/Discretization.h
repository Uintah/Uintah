
#ifndef Uintah_Components_Arches_Discretization_h
#define Uintah_Components_Arches_Discretization_h

//#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
//#include <Packages/Uintah/Core/Grid/CCVariable.h>
//#include <Packages/Uintah/Core/Grid/FCVariable.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>
#ifdef PetscFilter
#include <Packages/Uintah/CCA/Components/Arches/Filter.h>
#endif

#include <Core/Containers/Array1.h>
namespace Uintah {
  class ProcessorGroup;

//class StencilMatrix;
using namespace SCIRun;

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

class Discretization {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a Discretization.
      // PRECONDITIONS
      // POSTCONDITIONS
      // Default constructor.
      Discretization();

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~Discretization();

      // GROUP:  Action Methods
      ////////////////////////////////////////////////////////////////////////
      // Set stencil weights. (Velocity)
      // It uses second order hybrid differencing for computing
      // coefficients
      void calculateVelocityCoeff(const ProcessorGroup*,
				  const Patch* patch,
				  double delta_t,
				  int index, bool lcentral,
				  CellInformation* cellinfo,
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars);

      void calculateVelRhoHat(const ProcessorGroup* /*pc*/,
			      const Patch* patch,
			      int index, double delta_t,
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars);
      ////////////////////////////////////////////////////////////////////////
      // Set stencil weights. (Pressure)
      // It uses second order hybrid differencing for computing
      // coefficients
      void calculatePressureCoeff(const ProcessorGroup*,
				  const Patch* patch,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw,
				  double delta_t, 
				  CellInformation* cellinfo,
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars); 

      ////////////////////////////////////////////////////////////////////////
      // Modify stencil weights (Pressure) to account for voidage due
      // to multiple materials
      void mmModifyPressureCoeffs(const ProcessorGroup*,
				      const Patch* patch,
				      ArchesVariables* vars,
				      ArchesConstVariables* constvars);

      ////////////////////////////////////////////////////////////////////////
      // Set stencil weights. (Scalars)
      // It uses second order hybrid differencing for computing
      // coefficients
      void calculateScalarCoeff(const ProcessorGroup*,
				const Patch* patch,
				double delta_t,
				int Index,
				CellInformation* cellinfo,
				ArchesVariables* vars,
				ArchesConstVariables* constvars,
				int conv_scheme);

      ////////////////////////////////////////////////////////////////////////
      // Documentation here
      void calculateVelDiagonal(const ProcessorGroup*,
				const Patch* patch,
				int index,
				ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Documentation here
      void calculatePressDiagonal(const ProcessorGroup*,
				  const Patch* patch,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw, ArchesVariables* vars);

      ////////////////////////////////////////////////////////////////////////
      // Documentation here
      void calculateScalarDiagonal(const ProcessorGroup*,
				   const Patch* patch,
				   int Index, ArchesVariables* vars);
      ////////////////////////////////////////////////////////////////////////
      // Documentation here
      void calculateScalarENOscheme(const ProcessorGroup*,
				   const Patch* patch,
				   int Index,
				   CellInformation* cellinfo,
				   const double maxAbsU, const double maxAbsV,
				   const double maxAbsW,
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars,
				   const int wall_celltypeval);


      void calculateScalarWENOscheme(const ProcessorGroup*,
				   const Patch* patch,
				   int Index,
				   CellInformation* cellinfo,
				   const double maxAbsU, const double maxAbsV,
				   const double maxAbsW,
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars,
				   const int wall_celltypeval);


      void computeDivergence(const ProcessorGroup*,
				  const Patch* patch,
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars);

#ifdef PetscFilter
      inline void setFilter(Filter* filter) {
	d_filter = filter;
      }
#endif

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

#ifdef PetscFilter
      Filter* d_filter;
#endif
}; // end class Discretization

} // End namespace Uintah

#endif  
  
