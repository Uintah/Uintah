//
// $Id$
//

/**************************************
CLASS
   Discretization
   
   Class Discretization is an abstract base class
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

#ifndef included_Discretization
#define included_Discretization

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

class Discretization
{
public:
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  //
  // Construct an instance of a Discretization.
  //
  // PRECONDITIONS
  //
  //
  // POSTCONDITIONS
  //
  // Default constructor.
 
   Discretization();


  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual Destructor
   virtual ~Discretization();

   // GROUP:  Methods
   ////////////////////////////////////////////////////////////////////////
   // Set stencil weights. It uses second order hybrid differencing for computing
   // coefficients
    void calculateVelocityCoeff(const ProcessorContext*,
				const Region* region,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				double delta_t,
				const int Index);
    void calculatePressureCoeff(const ProcessorContext*,
				const Region* region,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				double delta_t); 
    void calculateScalarCoeff(const ProcessorContext*,
			      const Region* region,
			      const DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw,
			      double delta_t,
			      const int Index);

    void calculateVelDiagonal(const ProcessorContext*,
			      const Region* region,
			      const DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw,
			      const int Index);
    void calculatePressDiagonal(const ProcessorContext*,
				const Region* region,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw);
    void calculateScalarDiagonal(const ProcessorContext*,
				 const Region* region,
				 const DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw,
				 const int Index);
 private:
   

  // Stencil weights.
   // Array of size NDIM and of depth determined by stencil coefficients

   StencilMatrix* d_press_stencil_matrix;
   // stores coefficients for all the velocity components
   // coefficients should be saved on staggered grid
   StencilMatrix* d_mom_stencil_matrix;
   // coefficients for all the scalar components
   StencilMatrix* d_scalar_stencil_matrix;

};

}
}
#endif  
  
