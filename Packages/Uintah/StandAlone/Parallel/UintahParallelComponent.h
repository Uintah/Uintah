#ifndef Uintah_Parallel_UintahParallelCore/CCA/Component_h
#define Uintah_Parallel_UintahParallelCore/CCA/Component_h

#include <string>
#include <map>
#include <vector>

namespace Uintah {
   using std::string;
   class Packages/UintahParallelPort;
   class ProcessorGroup;

/**************************************

CLASS
   UintahParallelCore/CCA/Component
   
   Short description...

GENERAL INFORMATION

   UintahParallelCore/CCA/Component.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Uintah_Parallel_Core/CCA/Component, Core/CCA/Component

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class UintahParallelCore/CCA/Component {
      struct PortRecord {
	 PortRecord(Packages/UintahParallelPort* conn);
	 std::vector<Packages/UintahParallelPort*> connections;
      };
      std::map<string, PortRecord*> portmap;
   public:
      UintahParallelCore/CCA/Component(const ProcessorGroup* myworld);
      virtual ~UintahParallelCore/CCA/Component();
      
      //////////
      // Insert Documentation Here:
      void attachPort(const string& name, Packages/UintahParallelPort* port);
      
      Packages/UintahParallelPort* getPort(const std::string& name);
      void releasePort(const std::string& name);
      
   protected:
      const ProcessorGroup* d_myworld;
   };
} // End namespace Uintah
   


#endif
