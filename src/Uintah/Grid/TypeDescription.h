#ifndef UINTAH_HOMEBREW_TypeDescription_H
#define UINTAH_HOMEBREW_TypeDescription_H

namespace Uintah {
   
   /**************************************
     
     CLASS
       TypeDescription
      
       Short Description...
      
     GENERAL INFORMATION
      
       TypeDescription.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       TypeDescription
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class TypeDescription {
   public:
      enum Basis {
	 None,
	 Cell,
	 Node,
	 Particle
      };

     enum Type { 
       Scalar,
       Point,
       Vector,
       Tensor,
       Other 
     };


     TypeDescription(bool reductionvar, Basis basis,
		     Type type = TypeDescription::Other);
     
     bool isReductionVariable() const {
       return d_reductionvar;
     }
     Basis getBasis() const {
       return d_basis;
     }
     Type getType() const {
       return d_type;
     }

     private:
       bool d_reductionvar;
       Basis d_basis;
       Type d_type;
       ~TypeDescription();
       
       TypeDescription(const TypeDescription&);
       TypeDescription& operator=(const TypeDescription&);
       
     };
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/05/18 18:41:14  kuzimmer
// Added Particle to Basis enum, created Type enum with Scalar,Point,Vector,Tensor,& Other
//
// Revision 1.1  2000/05/07 06:02:13  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
//

#endif

