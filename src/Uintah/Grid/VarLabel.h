
#ifndef UINTAH_HOMEBREW_VarLabel_H
#define UINTAH_HOMEBREW_VarLabel_H

#include <string>
#include <iostream>
#include <map>

using std::ostream;
using std::string;

namespace Uintah {
   class TypeDescription;
   class Patch;
    
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
     
      VarLabel(const string&, const TypeDescription*,
	       VarType vartype = Normal);

      ~VarLabel();

      // VarLabel(const string&, const TypeDescription*);
      
      inline const string& getName() const {
	 return d_name;
      }
      string getFullName(int matlIndex, const Patch* patch) const;
      bool isPositionVariable() const {
	 return d_vartype == PositionVariable;
      }

      const TypeDescription* typeDescription() const {
	 return d_td;
      }

      static VarLabel* find(string name);
     
      class Compare {
      public:
	 inline bool operator()(const VarLabel* v1, const VarLabel* v2) const {
	    if(v1 == v2)
	       return false;
	    return v1->getName() < v2->getName();
	 }
      private:
      };

      string                 d_name;
   private:
      const TypeDescription* d_td;
      VarType                d_vartype;
      static std::map<std::string, VarLabel*> allLabels;     
      
      VarLabel(const VarLabel&);
      VarLabel& operator=(const VarLabel&);
   };
} // end namespace Uintah

ostream & operator<<( ostream & out, const Uintah::VarLabel & vl );

//
// $Log$
// Revision 1.14  2000/12/23 00:35:23  witzel
// Added a static member variable to VarLabel that maps VarLabel names to
// the appropriate VarLabel* for all VarLabel's in existent, and added
// VarLabel::find which uses this map.
//
// Revision 1.13  2000/12/10 09:06:18  sparker
// Merge from csafe_risky1
//
// Revision 1.12.4.1  2000/10/10 05:28:08  sparker
// Added support for NullScheduler (used for profiling taskgraph overhead)
//
// Revision 1.12  2000/09/26 21:39:22  dav
// inlined some things
//
// Revision 1.11  2000/09/25 20:37:43  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.10  2000/09/25 18:12:20  sparker
// do not use covariant return types due to problems with g++
// other linux/g++ fixes
//
// Revision 1.9  2000/08/23 22:36:50  dav
// added output operator
//
// Revision 1.8  2000/07/27 22:39:51  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.7  2000/05/07 06:02:14  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.6  2000/05/02 06:07:23  sparker
// Implemented more of DataWarehouse and SerialMPM
//
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

