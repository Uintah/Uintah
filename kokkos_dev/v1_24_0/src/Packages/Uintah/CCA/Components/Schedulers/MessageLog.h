#ifndef UINTAH_HOMEBREW_MessageLog_H
#define UINTAH_HOMEBREW_MessageLog_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <fstream>

namespace Uintah {
  class ProcessorGroup;
  class Output;
  class Patch;
  class VarLabel;

   /**************************************
     
     CLASS
       MessageLog
      
       Short Description...
      
     GENERAL INFORMATION
      
       MessageLog.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       MessageLog
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class MessageLog {
   public:
      MessageLog(const ProcessorGroup* myworld, Output* oport);
      void problemSetup(const ProblemSpecP& prob_spec);
      ~MessageLog();
#if 0
      void logSend(const DetailedReq* dep, int bytes,
		   const char* msg = 0);
      void logRecv(const DetailedReq* dep, int bytes,
		   const char* msg = 0);
#endif

      void finishTimestep();
   private:
      bool d_enabled;
      const ProcessorGroup* d_myworld;
      Output* d_oport;
      std::ofstream out;
      
      MessageLog(const MessageLog&);
      MessageLog& operator=(const MessageLog&);
      
   };
} // End namespace Uintah


#endif

