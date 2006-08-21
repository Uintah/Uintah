#ifndef UINTAH_HOMEBREW_ModelMaker_H
#define UINTAH_HOMEBREW_ModelMaker_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Util/Handle.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/CCA/Ports/share.h>

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
   class SCISHARE ModelMaker : public UintahParallelPort {
   public:
     ModelMaker();
     virtual ~ModelMaker();
           
     virtual std::vector<ModelInterface*> getModels() = 0;
     //////////
     // Insert Documentation Here:
     virtual void makeModels(const ProblemSpecP& restart_ps, 
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
