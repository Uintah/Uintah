#ifndef UINTAH_HOMEBREW_Variable_H
#define UINTAH_HOMEBREW_Variable_H

#include <string>
#include <iosfwd>
class DOM_Element;

namespace Uintah {

   using namespace std;
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

  void emit(OutputContext&, const string& compressionModeHint);
  void read(InputContext&, long end, const string& compressionMode);

  virtual void emitNormal(ostream& out, DOM_Element varnode) = 0;
  virtual void readNormal(istream& in) = 0;

  virtual bool emitRLE(ostream& /*out*/, DOM_Element /*varnode*/);
  virtual void readRLE(istream& /*in*/);
  
  virtual void allocate(const Patch* patch) = 0;

  void setAllocationLabel(const VarLabel* label)
  { allocationLabel_ = label; }

  const VarLabel* getAllocationLabel() const
  { return allocationLabel_; }
  
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
  string* gzipCompress(string* pUncompressed, string* pBuffer);
  bool d_foreign;
  const VarLabel* allocationLabel_;
};

} // End namespace Uintah

#endif
