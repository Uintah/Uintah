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
      UintahParallelPort* getPort(const std::string& name, unsigned int i);
      void releasePort(const std::string& name);
      unsigned int numConnections(const std::string& name);
      
   protected:
      const ProcessorGroup* d_myworld;
   };
} // End namespace Uintah
   
#endif
