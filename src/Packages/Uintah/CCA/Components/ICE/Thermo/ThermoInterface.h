#ifndef Uintah_ThermoInterface_h
#define Uintah_ThermoInterface_h

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/ICE/PropertyBase.h>

namespace Uintah {

/**************************************

CLASS
   ThermoInterface
   
   Short description...

GENERAL INFORMATION

   ThermoInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS


DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class ThermoInterface : public PropertyBase {
  public:
    ThermoInterface();
    virtual ~ThermoInterface();

    virtual void addTaskDependencies_thermalDiffusivity(Task* t, Task::WhichDW dw,
                                                        int numGhostCells) = 0;
    virtual void addTaskDependencies_Cp(Task* t, Task::WhichDW dw,
                                        int numGhostCells) = 0;
    virtual void addTaskDependencies_Cv(Task* t, Task::WhichDW dw,
                                        int numGhostCells) = 0;
    virtual void addTaskDependencies_gamma(Task* t, Task::WhichDW dw,
                                           int numGhostCells) = 0;
    virtual void addTaskDependencies_R(Task* t, Task::WhichDW dw,
                                       int numGhostCells) = 0;
  };
} // End namespace Uintah
      
#endif  // Uintah_ThermoInterface_h


