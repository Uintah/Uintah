#ifndef UINTAH_HOMEBREW_Variable_H
#define UINTAH_HOMEBREW_Variable_H

#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {

class TypeDescription;

/**************************************
     
  CLASS
    Variable

    Short Description...

  GENERAL INFORMATION

    Variable.h

    Steven G. Parker
    Department of Computer Science
    University of Utah
      
    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
    Copyright (C) 2000 SCI Group
      
  KEYWORDS
    Variable
      
  DESCRIPTION
    Long description...
      
  WARNING
      
  ****************************************/
    
class Variable {

public:
  virtual const TypeDescription* virtualGetTypeDescription() const = 0;
  void setForeign();
  bool isForeign() const {
    return d_foreign;
  }

  virtual void emit(OutputContext&) = 0;
  virtual void read(InputContext&) = 0;
  virtual void allocate(const Patch* patch) = 0;

protected:
  Variable();
  virtual ~Variable();

private:
  Variable(const Variable&);
  Variable& operator=(const Variable&);

  bool d_foreign;
};

} // End namespace Uintah

#endif
