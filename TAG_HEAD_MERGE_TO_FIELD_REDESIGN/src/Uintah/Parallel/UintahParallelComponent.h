#ifndef Uintah_Parallel_UintahParallelComponent_h
#define Uintah_Parallel_UintahParallelComponent_h

#include <string>
#include <map>
#include <vector>

namespace Uintah {

   using std::string;
   class UintahParallelPort;
   class ProcessorGroup;

/**************************************

CLASS
   UintahParallelComponent
   
   Short description...

GENERAL INFORMATION

   UintahParallelComponent.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Uintah_Parallel_Component, Component

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class UintahParallelComponent {
      struct PortRecord {
	 PortRecord(UintahParallelPort* conn);
	 std::vector<UintahParallelPort*> connections;
      };
      std::map<string, PortRecord*> portmap;
   public:
      UintahParallelComponent(const ProcessorGroup* myworld);
      virtual ~UintahParallelComponent();
      
      //////////
      // Insert Documentation Here:
      void attachPort(const string& name, UintahParallelPort* port);
      
      UintahParallelPort* getPort(const std::string& name);
      void releasePort(const std::string& name);
      
   protected:
      const ProcessorGroup* d_myworld;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.6  2000/06/17 07:06:50  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.5  2000/04/26 06:49:16  sparker
// Streamlined namespaces
//
// Revision 1.4  2000/04/19 21:20:05  dav
// more MPI stuff
//
// Revision 1.3  2000/04/11 07:10:57  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.2  2000/03/16 22:08:39  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
