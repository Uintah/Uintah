#ifndef UINTAH_GRID_BoundCond_H
#define UINTAH_GRID_BoundCond_H

#include <Packages/Uintah/Core/Grid/BoundCondBase.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
   
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
#if 0
      BoundCond(const BoundCond<T>&);
      BoundCond& operator=(const BoundCond<T>&);
#endif
      
     
   };

} // End namespace Uintah

#endif
