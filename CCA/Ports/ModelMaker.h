#ifndef UINTAH_HOMEBREW_ModelMaker_H
#define UINTAH_HOMEBREW_ModelMaker_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

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
      
     //////////
     // Insert Documentation Here:
     virtual void makeModels(const ProblemSpecP& params, GridP& grid,
			     SimulationStateP& sharedState,
			     std::vector<ModelInterface*>& models) = 0;
   private:
     ModelMaker(const ModelMaker&);
     ModelMaker& operator=(const ModelMaker&);
   };
} // End namespace Uintah
   


#endif
