#ifndef UINTAH_GRID_BoundCond_H
#define UINTAH_GRID_BoundCond_H

#include <string>


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

   class BoundCond  {
   public:
      BoundCond() {};
      virtual ~BoundCond() {};
      virtual std::string getType() const = 0;
         
   private:
      BoundCond(const BoundCond&);
      BoundCond& operator=(const BoundCond&);
      
     
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/27 22:31:49  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




