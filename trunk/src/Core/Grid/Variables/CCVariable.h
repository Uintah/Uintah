#ifndef UINTAH_HOMEBREW_CCVARIABLE_H
#define UINTAH_HOMEBREW_CCVARIABLE_H

#include <Core/Grid/Variables/GridVariable.h>
#include <Core/Grid/Variables/constGridVariable.h>
#include <Core/Grid/Patch.h>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace Uintah {

  using SCIRun::InternalError;

  class TypeDescription;

  /**************************************

CLASS
   CCVariable
   
   Short description...

GENERAL INFORMATION

   CCVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Variable__Cell_Centered

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class T> 
  class CCVariable : public GridVariable<T> {
    friend class constVariable<GridVariableBase, CCVariable<T>, T, const IntVector&>;
  public:
    CCVariable();
    virtual ~CCVariable();
      
    //////////
    // Insert Documentation Here:
    const TypeDescription* virtualGetTypeDescription() const 
    { return getTypeDescription(); }
    static const TypeDescription* getTypeDescription();
    
    virtual GridVariableBase* clone();
    virtual const GridVariableBase* clone() const;
    virtual GridVariableBase* cloneType() const
    { return scinew CCVariable<T>(); }
    
    // allocate(IntVector, IntVector) is hidden without this
    using GridVariable<T>::allocate;
    virtual void allocate(const Patch* patch, const IntVector& boundary)
    {      
      IntVector l,h;
      patch->computeVariableExtents(Patch::CellBased, boundary, 
                                    Ghost::None, 0, l, h);
      GridVariable<T>::allocate(l, h);
    }

    static TypeDescription::Register registerMe;

  protected:
    CCVariable(const CCVariable<T>&);

  private:
    CCVariable<T>& operator=(const CCVariable<T>&);

    static Variable* maker();
  };

  template<class T>
  TypeDescription::Register
  CCVariable<T>::registerMe(getTypeDescription());
   
  template<class T>
  const TypeDescription*
  CCVariable<T>::getTypeDescription()
  {
    static TypeDescription* td;
    if(!td){
      td = scinew TypeDescription(TypeDescription::CCVariable,
				  "CCVariable", &maker,
				  fun_getTypeDescription((T*)0));
    }
    return td;
  }
    
  template<class T>
  Variable*
  CCVariable<T>::maker()
  {
    return scinew CCVariable<T>();
  }
   
  template<class T>
  CCVariable<T>::~CCVariable()
  {
  }
   
  template<class T>
  GridVariableBase*
  CCVariable<T>::clone()
  {
    return scinew CCVariable<T>(*this);
  }

  template<class T>
  const GridVariableBase*
  CCVariable<T>::clone() const
  {
    return scinew CCVariable<T>(*this);
  }

  template<class T>
  CCVariable<T>::CCVariable()
  {
  }

  template<class T>
  CCVariable<T>::CCVariable(const CCVariable<T>& copy)
    : GridVariable<T>(copy)
  {
  }

  template <class T>
  class constCCVariable : public constGridVariable<GridVariableBase, CCVariable<T>, T>
  {
  public:
    constCCVariable() : constGridVariable<GridVariableBase, CCVariable<T>, T>() {}
    constCCVariable(const CCVariable<T>& copy) : constGridVariable<GridVariableBase, CCVariable<T>, T>(copy) {}
  };


} // end namespace Uintah

#endif

