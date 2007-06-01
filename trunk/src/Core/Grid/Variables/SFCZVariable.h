#ifndef UINTAH_HOMEBREW_SFCZVARIABLE_H
#define UINTAH_HOMEBREW_SFCZVARIABLE_H

#include <Core/Grid/Variables/GridVariable.h>
#include <Core/Grid/Variables/constGridVariable.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/InputContext.h>
#include <CCA/Ports/OutputContext.h>
#include <Core/IO/SpecializedRunLengthEncoder.h>
#include <Core/Exceptions/TypeMismatchException.h>

#include <SCIRun/Core/Exceptions/InternalError.h>
#include <SCIRun/Core/Geometry/Vector.h>
#include <SCIRun/Core/Malloc/Allocator.h>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace Uintah {

  using SCIRun::InternalError;

  class TypeDescription;

  /**************************************

CLASS
   SFCZVariable
   
   Short description...

GENERAL INFORMATION

   SFCZVariable.h

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
  class SFCZVariable : public GridVariable<T> {
    friend class constVariable<GridVariableBase, SFCZVariable<T>, T, const IntVector&>;
  public:
    SFCZVariable();
    virtual ~SFCZVariable();
      
    //////////
    // Insert Documentation Here:
    const TypeDescription* virtualGetTypeDescription() const 
    { return getTypeDescription(); }
    static const TypeDescription* getTypeDescription();
    
    virtual GridVariableBase* clone();
    virtual const GridVariableBase* clone() const;
    virtual GridVariableBase* cloneType() const
    { return scinew SFCZVariable<T>(); }
    
    // allocate(IntVector, IntVector) is hidden without this
    using GridVariable<T>::allocate;
    virtual void allocate(const Patch* patch, const IntVector& boundary)
    {      
      IntVector l,h;
      patch->computeVariableExtents(Patch::ZFaceBased, boundary, 
                                    Ghost::None, 0, l, h);
      GridVariable<T>::allocate(l, h);
    }

    static TypeDescription::Register registerMe;

  protected:
    SFCZVariable(const SFCZVariable<T>&);

  private:
    SFCZVariable<T>& operator=(const SFCZVariable<T>&);

    static Variable* maker();
  };

  template<class T>
  TypeDescription::Register
  SFCZVariable<T>::registerMe(getTypeDescription());
   
  template<class T>
  const TypeDescription*
  SFCZVariable<T>::getTypeDescription()
  {
    static TypeDescription* td;
    if(!td){
      td = scinew TypeDescription(TypeDescription::SFCZVariable,
				  "SFCZVariable", &maker,
				  fun_getTypeDescription((T*)0));
    }
    return td;
  }
    
  template<class T>
  Variable*
  SFCZVariable<T>::maker()
  {
    return scinew SFCZVariable<T>();
  }
   
  template<class T>
  SFCZVariable<T>::~SFCZVariable()
  {
  }
   
  template<class T>
  GridVariableBase*
  SFCZVariable<T>::clone()
  {
    return scinew SFCZVariable<T>(*this);
  }

  template<class T>
  const GridVariableBase*
  SFCZVariable<T>::clone() const
  {
    return scinew SFCZVariable<T>(*this);
  }

  template<class T>
  SFCZVariable<T>::SFCZVariable()
  {
  }

  template<class T>
  SFCZVariable<T>::SFCZVariable(const SFCZVariable<T>& copy)
    : GridVariable<T>(copy)
  {
  }
   
  template <class T>
  class constSFCZVariable : public constGridVariable<GridVariableBase, SFCZVariable<T>, T>
  {
  public:
    constSFCZVariable() : constGridVariable<GridVariableBase, SFCZVariable<T>, T>() {}
    constSFCZVariable(const SFCZVariable<T>& copy) : constGridVariable<GridVariableBase, SFCZVariable<T>, T>(copy) {}
  };

} // end namespace Uintah

#endif

