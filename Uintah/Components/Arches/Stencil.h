//
// $Id$
//

#ifndef Uintah_Components_Arches_Stencil_h
#define Uintah_Components_Arches_Stencil_h

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

namespace Uintah {
namespace ArchesSpace {

    class Stencil {
	Stencil(const DataWarehouseP&, const std::string&, const Patch*);
    };

}
}
#endif  
  
//
// $Log$
// Revision 1.5  2000/06/04 22:40:16  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
