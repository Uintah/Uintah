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
namespace Uintah {
namespace Components {

    class Stencil {
	Stencil(const DataWarehouseP&, const std::string&, const Region*);
    };

}
}
#endif  
  
