#ifndef UINTAH_HOMEBREW_TypeDescription_H
#define UINTAH_HOMEBREW_TypeDescription_H

#include <string>

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
      enum Type {
	 CCVariable,
	 NCVariable,
	 ParticleVariable,
	 Point,
	 Vector,
	 Matrix3,
	 ReductionVariable,
	 double_type,
	 bool_type,
	 int_type,
         long_type,
	 Unknown,
	 Other
      };

      TypeDescription(Type type, const std::string& name,
		      bool isFlat);
      TypeDescription(Type type, const std::string& name,
		      const TypeDescription* subtype);
     
      bool isReductionVariable() const {
	 return d_type == ReductionVariable;
      }
      Type getType() const {
	 return d_type;
      }
      const TypeDescription* getSubType() const {
	 return d_subtype;
      }
      std::string getName() const;

      bool isFlat() const {
	 return d_isFlat;
      }

      struct Register {
	 Register(const TypeDescription*);
	 ~Register();
      };
      static const TypeDescription* lookupType(const std::string&);
   private:
      Type d_type;
      const TypeDescription* d_subtype;
      std::string d_name;
      bool d_isFlat;
      ~TypeDescription();
       
      TypeDescription(const TypeDescription&);
      TypeDescription& operator=(const TypeDescription&);
       
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/06/02 17:22:14  guilkey
// Added long_type to the the TypeDescription and TypeUtils.
//
// Revision 1.4  2000/05/21 08:19:09  sparker
// Implement NCVariable read
// Do not fail if variable type is not known
// Added misc stuff to makefiles to remove warnings
//
// Revision 1.3  2000/05/20 08:09:29  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.2  2000/05/18 18:41:14  kuzimmer
// Added Particle to Basis enum, created Type enum with Scalar,Point,Vector,Tensor,& Other
//
// Revision 1.1  2000/05/07 06:02:13  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
//

#endif

