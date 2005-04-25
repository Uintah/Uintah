#ifndef UINTAH_HOMEBREW_CONSTVARIABLEBASE_H
#define UINTAH_HOMEBREW_CONSTVARIABLEBASE_H


namespace Uintah {

  class TypeDescription;

  /**************************************

CLASS
   constVariableBase
   
   Version of *VariableBase that is const in the sense that you can't
   modify the data that it points to (although you can change what it
   points to if it is a non-const version of the constVariableBase).

GENERAL INFORMATION

   constVariableBase.h

   Wayne Witzel
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2001 SCI Group

KEYWORDS
   Variable, const

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class VariableBase> 
  class constVariableBase {
  public:   
    virtual ~constVariableBase() {}

    virtual constVariableBase& operator=(const VariableBase&) = 0;
    virtual constVariableBase& operator=(const constVariableBase&) = 0;

    operator const VariableBase&() const
    { return getBaseRep(); }
   
    virtual const VariableBase& getBaseRep() = 0;
    
    virtual void copyPointer(const VariableBase& copy) = 0;

    virtual const VariableBase* clone() const = 0;
    virtual VariableBase* cloneType() const = 0;

    virtual const TypeDescription* virtualGetTypeDescription() const = 0;

    /*
    virtual void getSizes(IntVector& low, IntVector& high,
			  IntVector& dataLow, IntVector& siz,
			  IntVector& strides) const = 0;

    virtual void getSizeInfo(string& elems, unsigned long& totsize,
			     void*& ptr) const = 0;
    */
  protected:
    constVariableBase() {}
    constVariableBase(const constVariableBase&) {}
  private:
  };

} // end namespace Uintah


#endif

