#ifndef UINTAH_HOMEBREW_Variable_H
#define UINTAH_HOMEBREW_Variable_H

#include <Packages/Uintah/Core/Exceptions/InvalidCompressionMode.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SpecializedRunLengthEncoder.h>
#include <iostream>

namespace Uintah {

   using namespace std;
   class TypeDescription;
   class InputContext;
   class OutputContext;
   class Patch;

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

  void emit(OutputContext&, string compressionMode);
  void read(InputContext&, long end, string compressionMode);

  virtual void emitNormal(ostream& out, DOM_Element varnode) = 0;
  virtual void readNormal(istream& in) = 0;

  virtual void emitRLE(ostream& /*out*/, DOM_Element /*varnode*/ /*,
				  bool *//*equal_only*/ /* can force it to use
					      EqualElementSequencer */)
  { throw InvalidCompressionMode("rle",
				 virtualGetTypeDescription()->getName()); }
  
  virtual void readRLE(istream& /*in*/ /*,
				  bool*/ /*equal_only*/ /* can force it to use
			                      EqualElementSequencer */)
  { throw InvalidCompressionMode("rle",
				 virtualGetTypeDescription()->getName()); }
  
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
