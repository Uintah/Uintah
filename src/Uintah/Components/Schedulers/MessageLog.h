#ifndef UINTAH_HOMEBREW_MessageLog_H
#define UINTAH_HOMEBREW_MessageLog_H

#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/Task.h>
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
      void logSend(const Task::Dependency* dep, int bytes,
		   const char* msg = 0);
      void logRecv(const Task::Dependency* dep, int bytes,
		   const char* msg = 0);

      void finishTimestep();
   private:
      bool d_enabled;
      const ProcessorGroup* d_myworld;
      Output* d_oport;
      std::ofstream out;
      
      MessageLog(const MessageLog&);
      MessageLog& operator=(const MessageLog&);
      
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/09/20 16:00:28  sparker
// Added external interface to LoadBalancer (for per-processor tasks)
// Added message logging functionality. Put the tag <MessageLog/> in
//    the ups file to enable
//
//

#endif

