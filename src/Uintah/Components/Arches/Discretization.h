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
namespace Components {
class StencilMatrix;
using namespace Uintah::Grid;
  using namespace Uintah::Interface;
  using namespace SCICore::Containers;
  using namespace Uintah::Parallel;

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
   virtual void sched_calculatePressureCoeff(const LevelP& level,
					     SchedulerP& sched,
					     const DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw);
   virtual void sched_calculateVelocityCoeff(const int index,
					     const LevelP& level,
					     SchedulerP& sched,
					     const DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw);
   virtual void sched_calculateScalarCoeff(const int index,
					   const LevelP& level,
					   SchedulerP& sched,
					   const DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw);

 private:
   
    void calculateVelocityCoeff(const ProcessorContext*,
			  const Region* region,
			  const DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw,
				const int Index);
    void calculatePressureCoeff(const ProcessorContext*,
				const Region* region,
				const DataWarehouseP& old_dw,
				DataWarehouseP& new_dw); 

  // Stencil weights.
   // Array of size NDIM and of depth determined by stencil coefficients

   StencilMatrix* d_press_stencil_matrix;
   // stores coefficients for all the velocity components
   // coefficients should be saved on staggered grid
   StencilMatrix* d_mom_stencil_matrix;
   // coefficients for all the scalar components
   StencilMatrix* d_scalar_stencil_matrix;

   struct CellInformation {
     // p-cell geom information
     // x- direction
     Array1<double> dxep;
     Array1<double> dxpw;
     Array1<double> sew;
     // y- direction
     Array1<double> dynp;
     Array1<double> dyps;
     Array1<double> sns;
     // z-direction
     Array1<double> dztp;
     Array1<double> dzpb;
     Array1<double> stb;
     //u-cell geom info
     Array1<double> xu;
     Array1<double> dxepu;
     Array1<double> dxpwu;
     Array1<double> sewu;
     //v-cell geom info
     Array1<double> yv;
     Array1<double> dynpv;
     Array1<double> dypsv;
     Array1<double> snsv;
     //w-cell geom info
     Array1<double> zw;
     Array1<double> dztpw;
     Array1<double> dzpbw;
     Array1<double> stbw;
     //differencing factors for p-cell
     // x-direction
     Array1<double> cee;
     Array1<double> cww;
     Array1<double> cwe;
     Array1<double> ceeu;
     Array1<double> cwwu;
     Array1<double> cweu;
     Array1<double> efac;
     Array1<double> wfac;
     // y-direction
     Array1<double> cnn;
     Array1<double> css;
     Array1<double> csn;
     Array1<double> cnnv;
     Array1<double> cssv;
     Array1<double> csnv;
     Array1<double> enfac;
     Array1<double> sfac;
     // z-direction
     Array1<double> ctt;
     Array1<double> cbb;
     Array1<double> cbt;
     Array1<double> cttw;
     Array1<double> cbbw;
     Array1<double> cbtw;
     Array1<double> tfac;
     Array1<double> bfac;
     // factors for differencing u-cell
     Array1<double> fac1u;
     Array1<double> fac2u;
     Array1<double> iesdu;
     Array1<double> fac3u;
     Array1<double> fac4u;
     Array1<double> iwsdu;
     // factors for differencing v-cell
     Array1<double> fac1v;
     Array1<double> fac2v;
     Array1<double> jnsdv;
     Array1<double> fac3v;
     Array1<double> fac4v;
     Array1<double> jssdv;
     // factors for differencing w-cell
     Array1<double> fac1w;
     Array1<double> fac2w;
     Array1<double> ktsdw;
     Array1<double> fac3w;
     Array1<double> fac4w;
     Array1<double> kbsdw;
     // constructor computes the values
     CellInformation(const Region*);
   };
};

}
}
#endif  
  
