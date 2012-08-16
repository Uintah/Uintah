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


#ifndef UINTAH_HOMEBREW_ModelMaker_H
#define UINTAH_HOMEBREW_ModelMaker_H

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/GridP.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <vector>


namespace Uintah {
/**************************************

CLASS
   ModelMaker
   
   Short description...

GENERAL INFORMATION

   ModelMaker.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Model of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Model_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class ModelInterface;
   class ModelMaker : public UintahParallelPort {
   public:
     ModelMaker();
     virtual ~ModelMaker();
           
     virtual std::vector<ModelInterface*> getModels() = 0;
     virtual void clearModels() = 0;
     //////////
     // Insert Documentation Here:
     virtual void makeModels(const ProblemSpecP& orig_or_restart_ps, 
                             const ProblemSpecP& prob_spec,
                             GridP& grid,
                             SimulationStateP& sharedState,
                             const bool doAMR) = 0;

     virtual void outputProblemSpec(ProblemSpecP& ps) = 0;

   private:
     ModelMaker(const ModelMaker&);
     ModelMaker& operator=(const ModelMaker&);
   };
} // End namespace Uintah
   


#endif
