/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_MessageLog_H
#define UINTAH_HOMEBREW_MessageLog_H

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Task.h>
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

