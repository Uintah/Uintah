
#ifndef UINTAH_HOMEBREW_VarLabel_H
#define UINTAH_HOMEBREW_VarLabel_H

#include <string>

namespace Uintah {
   class TypeDescription;
    
    /**************************************
      
      CLASS
        VarLabel
      
        Short Description...
      
      GENERAL INFORMATION
      
        VarLabel.h
      
        Steven G. Parker
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
        Copyright (C) 2000 SCI Group
      
      KEYWORDS
        VarLabel
      
      DESCRIPTION
        Long description...
      
      WARNING
      
      ****************************************/
    
   class VarLabel {
   public:
      enum VarType {
	 Normal,
	 Internal,
	 PositionVariable
      };
     
      VarLabel(const std::string&, const TypeDescription*,
	       VarType vartype = Normal);

      // VarLabel(const std::string&, const TypeDescription*);
      
      const std::string& getName() const {
	 return d_name;
      }
      bool isPositionVariable() const {
	 return d_vartype == PositionVariable;
      }
   private:
      std::string d_name;
      const TypeDescription* d_td;
      VarType d_vartype;
      
      VarLabel(const VarLabel&);
      VarLabel& operator=(const VarLabel&);
   };
   
    
} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/04/28 20:24:44  jas
// Moved some private copy constructors to public for linux.  Velocity
// field is now set from the input file.  Simulation state now correctly
// determines number of velocity fields.
//
// Revision 1.4  2000/04/28 07:35:37  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.3  2000/04/26 06:49:01  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/04/20 18:56:32  sparker
// Updates to MPM
//
// Revision 1.1  2000/04/19 05:26:15  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
//

#endif

