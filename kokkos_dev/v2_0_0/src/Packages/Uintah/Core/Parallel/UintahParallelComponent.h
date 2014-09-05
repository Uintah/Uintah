#ifndef Uintah_Parallel_UintahParallelComponent_h
#define Uintah_Parallel_UintahParallelComponent_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <map>
#include <vector>
#include <sgi_stl_warnings_on.h>

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
   Uintah_Parallel_Dataflow/Component, Dataflow/Component

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
} // End namespace Uintah
   
#endif
