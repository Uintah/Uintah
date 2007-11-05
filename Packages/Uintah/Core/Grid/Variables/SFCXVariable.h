#ifndef UINTAH_HOMEBREW_SFCXVARIABLE_H
#define UINTAH_HOMEBREW_SFCXVARIABLE_H

#include <Packages/Uintah/Core/Grid/Variables/GridVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/constGridVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace Uintah {

  using SCIRun::InternalError;

  class TypeDescription;

  /**************************************

CLASS
   SFCXVariable
   
   Short description...

GENERAL INFORMATION

   SFCXVariable.h

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
  class SFCXVariable : public GridVariable<T> {
    friend class constVariable<GridVariableBase, SFCXVariable<T>, T, const IntVector&>;
  public:
    SFCXVariable();
    virtual ~SFCXVariable();
      
    //////////
    // Insert Documentation Here:
    const TypeDescription* virtualGetTypeDescription() const 
    { return getTypeDescription(); }
    static const TypeDescription* getTypeDescription();
    
    virtual GridVariableBase* clone();
    virtual const GridVariableBase* clone() const;
    virtual GridVariableBase* cloneType() const
    { return scinew SFCXVariable<T>(); }

    // allocate(IntVector, IntVector) is hidden without this
    using GridVariable<T>::allocate;
    virtual void allocate(const Patch* patch, const IntVector& boundary)
    {      
      IntVector l,h;
      patch->computeVariableExtents(Patch::XFaceBased, boundary, 
                                    Ghost::None, 0, l, h);
      GridVariable<T>::allocate(l, h);
    }

    static TypeDescription::Register registerMe;

  protected:
    SFCXVariable(const SFCXVariable<T>&);

  private:
    SFCXVariable<T>& operator=(const SFCXVariable<T>&);

    static Variable* maker();
  };

  template<class T>
  TypeDescription::Register
  SFCXVariable<T>::registerMe(getTypeDescription());
   
  template<class T>
  const TypeDescription*
  SFCXVariable<T>::getTypeDescription()
  {
    static TypeDescription* td;
    if(!td){
      td = scinew TypeDescription(TypeDescription::SFCXVariable,
				  "SFCXVariable", &maker,
				  fun_getTypeDescription((T*)0));
    }
    return td;
  }
    
  template<class T>
  Variable*
  SFCXVariable<T>::maker()
  {
    return scinew SFCXVariable<T>();
  }
   
  template<class T>
  SFCXVariable<T>::~SFCXVariable()
  {
  }
   
  template<class T>
  GridVariableBase*
  SFCXVariable<T>::clone()
  {
    return scinew SFCXVariable<T>(*this);
  }

  template<class T>
  const GridVariableBase*
  SFCXVariable<T>::clone() const
  {
    return scinew SFCXVariable<T>(*this);
  }

  template<class T>
  SFCXVariable<T>::SFCXVariable()
  {
  }

  template<class T>
  SFCXVariable<T>::SFCXVariable(const SFCXVariable<T>& copy)
    : GridVariable<T>(copy)
  {
  }
   
  template <class T>
  class constSFCXVariable : public constGridVariable<GridVariableBase, SFCXVariable<T>, T>
  {
  public:
    constSFCXVariable() : constGridVariable<GridVariableBase, SFCXVariable<T>, T>() {}
    constSFCXVariable(const SFCXVariable<T>& copy) : constGridVariable<GridVariableBase, SFCXVariable<T>, T>(copy) {}
  };

} // end namespace Uintah

#endif

