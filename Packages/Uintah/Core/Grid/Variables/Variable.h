#ifndef UINTAH_HOMEBREW_Variable_H
#define UINTAH_HOMEBREW_Variable_H

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

namespace SCIRun {
  class IntVector;
}

namespace Uintah {

  class TypeDescription;
  class InputContext;
  class OutputContext;
  class Patch;
  class RefCounted;
  class VarLabel;

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
  virtual ~Variable();
  
  virtual const TypeDescription* virtualGetTypeDescription() const = 0;
  void setForeign();
  bool isForeign() const {
    return d_foreign;
  }

  void emit(OutputContext&, const IntVector& l, const IntVector& h,
	    const string& compressionModeHint);
  void read(InputContext&, long end, bool swapbytes, int nByteMode,
	    const string& compressionMode);

  virtual void emitNormal(std::ostream& out, const IntVector& l,
			  const IntVector& h, ProblemSpecP varnode, bool outputDoubleAsFloat ) = 0;
  virtual void readNormal(std::istream& in, bool swapbytes) = 0;

  virtual bool emitRLE(std::ostream& /*out*/, const IntVector& l,
		       const IntVector& h, ProblemSpecP /*varnode*/);
  virtual void readRLE(std::istream& /*in*/, bool swapbytes, int nByteMode);
  
  virtual void allocate(const Patch* patch, const IntVector& boundary) = 0;

  virtual void getSizeInfo(std::string& elems, unsigned long& totsize, void*& ptr) const = 0;

  virtual void copyPointer(Variable&) = 0;

  // Only affects grid variables
  virtual void offsetGrid(const IntVector& /*offset*/);

  virtual RefCounted* getRefCounted() = 0;
protected:
  Variable();

private:    
  Variable(const Variable&);
  Variable& operator=(const Variable&);

  // Compresses the string pointed to by pUncompressed and but the
  // resulting compressed data into the string pointed to by pBuffer.
  // Returns the pointer to whichever one is shortest and erases the
  // other one.
  std::string* gzipCompress(std::string* pUncompressed, std::string* pBuffer);
  bool d_foreign;
};

} // End namespace Uintah

#endif
