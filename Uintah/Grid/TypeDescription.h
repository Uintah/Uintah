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
	 Node
      };
      TypeDescription(bool reductionvar, Basis basis);

      bool isReductionVariable() const {
	 return d_reductionvar;
      }
      Basis getBasis() const {
	 return d_basis;
      }

   private:
      bool d_reductionvar;
      Basis d_basis;
      ~TypeDescription();

      TypeDescription(const TypeDescription&);
      TypeDescription& operator=(const TypeDescription&);
      
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/05/07 06:02:13  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
//

#endif

