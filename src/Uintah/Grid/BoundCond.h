#ifndef UINTAH_GRID_BoundCond_H
#define UINTAH_GRID_BoundCond_H

#include <Uintah/Grid/BoundCondBase.h>
#include <string>
using std::string;

namespace Uintah {

   
/**************************************

CLASS
   BoundCond
   
   
GENERAL INFORMATION

   BoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template <class T>  class BoundCond : public BoundCondBase {
   public:
     BoundCond() {};
     BoundCond(const string& kind) 
       {
	 BoundCondBase();
	 d_kind=kind;
     };
     virtual ~BoundCond() {};
     string getKind() const 
       {
	 // Tells whether it is Dirichlet or Neumann
	 return d_kind;
       };
     virtual T getValue() const = 0; 

   protected:
	std::string d_kind;
         
   private:
      BoundCond(const BoundCond<T>&);
      BoundCond& operator=(const BoundCond<T>&);
      
     
   };



   
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/11/02 21:25:55  jas
// Rearranged the boundary conditions so there is consistency between ICE
// and MPM.  Added fillFaceFlux for the Neumann BC condition.  BCs are now
// declared differently in the *.ups file.
//
// Revision 1.1  2000/06/27 22:31:49  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




