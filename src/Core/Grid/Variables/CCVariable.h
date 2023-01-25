/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_CCVARIABLE_H
#define UINTAH_HOMEBREW_CCVARIABLE_H

#include <Core/Grid/Variables/GridVariable.h>
#include <Core/Grid/Variables/constGridVariable.h>
#include <Core/Grid/Patch.h>

namespace Uintah {

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
    const TypeDescription* virtualGetTypeDescription() const { return getTypeDescription(); }
    static const TypeDescription* getTypeDescription();
    
    virtual GridVariableBase* clone();
    virtual const GridVariableBase* clone() const;
    virtual GridVariableBase* cloneType() const { return scinew CCVariable<T>(); }
    
    // allocate(IntVector, IntVector) is hidden without this
    using GridVariable<T>::allocate;
    virtual void allocate(const Patch* patch, const IntVector& boundary)
    {      
      IntVector l,h;
      patch->computeVariableExtents(Patch::CellBased, boundary, 
                                    Ghost::None, 0, l, h);
      GridVariable<T>::allocate(l, h);
    }

    // Static variable whose entire purpose is to cause the (instantiated) type of this
    // class to be registered with the Core/Disclosure/TypeDescription class when this
    // class' object code is originally loaded from the shared library.  The 'registerMe'
    // variable is not used for anything else in the program.
    static TypeDescription::Register registerMe;

    CCVariable(CCVariable<T>&&);
    CCVariable(const CCVariable<T>&);

  protected:

  private:
    static TypeDescription* td;
    
    CCVariable<T>& operator=(const CCVariable<T>&);

    static Variable* maker();
  };

  template<class T>
  TypeDescription* CCVariable<T>::td = 0;

  // The following line is the initialization (creation) of the 'registerMe' static variable
  // (for each version of CCVariable (double, int, etc)).  Note, the 'registerMe' variable
  // is created when the object code is initially loaded (usually during intial program load
  // by the operating system).
  template<class T>
  TypeDescription::Register
  CCVariable<T>::registerMe( getTypeDescription() );
   
  template<class T>
  const TypeDescription*
  CCVariable<T>::getTypeDescription()
  {
    if( !td ){
      td = scinew TypeDescription( TypeDescription::CCVariable,
                                   "CCVariable", &maker,
                                   fun_getTypeDescription((T*)0) );
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
  CCVariable<T>::CCVariable(CCVariable<T>&& other)
    : GridVariable<T>(other)
  {
    // Implementing this somehow turned on and properly supported Return Value Optimization (RVO).  I'm not entirely sure why -- Brad P June 2018
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

