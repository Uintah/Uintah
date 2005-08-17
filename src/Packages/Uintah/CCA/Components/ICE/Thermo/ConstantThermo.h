#ifndef Uintah_ConstantThermo_h
#define Uintah_ConstantThermo_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/ThermoInterface.h>

namespace Uintah {

/**************************************

CLASS
   ConstantThermo
   
   Short description...

GENERAL INFORMATION

   ConstantThermo.h

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

  class ConstantThermo : public ThermoInterface {
  public:
    ConstantThermo(ProblemSpecP& ps);
    virtual ~ConstantThermo();

    virtual void addTaskDependencies_thermalDiffusivity(Task* t, Task::WhichDW dw,
                                                        int numGhostCells);
    virtual void addTaskDependencies_Cp(Task* t, Task::WhichDW dw,
                                        int numGhostCells);
    virtual void addTaskDependencies_Cv(Task* t, Task::WhichDW dw,
                                        int numGhostCells);
    virtual void addTaskDependencies_gamma(Task* t, Task::WhichDW dw,
                                           int numGhostCells);
    virtual void addTaskDependencies_R(Task* t, Task::WhichDW dw,
                                       int numGhostCells);
  private:
    double d_thermalConductivity;
    double d_specificHeat;
    double d_gamma;
  };
} // End namespace Uintah
      
#endif  // Uintah_ConstantThermo_h
