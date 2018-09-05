/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef UINTAH_HOMEBREW_UnstructuredSFCYVARIABLE_H
#define UINTAH_HOMEBREW_UnstructuredSFCYVARIABLE_H

#include <Core/Grid/Variables/UnstructuredGridVariable.h>
#include <Core/Grid/Variables/constUnstructuredGridVariable.h>
#include <Core/Grid/UnstructuredPatch.h>
namespace Uintah {

  class UnstructuredTypeDescription;

  /**************************************

CLASS
   UnstructuredSFCYVariable
   
   Short description...

GENERAL INFORMATION

   UnstructuredSFCYVariable.h

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
  class UnstructuredSFCYVariable : public UnstructuredGridVariable<T> {
    friend class constVariable<UnstructuredGridVariableBase, UnstructuredSFCYVariable<T>, T, const IntVector&>;
  public:
    UnstructuredSFCYVariable();
    virtual ~UnstructuredSFCYVariable();
      
    //////////
    // Insert Documentation Here:
    const UnstructuredTypeDescription* virtualGetUnstructuredTypeDescription() const 
    { return getUnstructuredTypeDescription(); }
    static const UnstructuredTypeDescription* getUnstructuredTypeDescription();
    

    virtual UnstructuredGridVariableBase* clone();
    virtual const UnstructuredGridVariableBase* clone() const;
    virtual UnstructuredGridVariableBase* cloneType() const
    { return scinew UnstructuredSFCYVariable<T>(); }
    
    // allocate(IntVector, IntVector) is hidden without this
    using UnstructuredGridVariable<T>::allocate;
    virtual void allocate(const UnstructuredPatch* patch, const IntVector& boundary)
    {      
      IntVector l,h;
      patch->computeVariableExtents(UnstructuredPatch::YFaceBased, boundary, 
                                    Ghost::None, 0, l, h);
      this->allocate(l, h);
    }

    // Static variable whose entire purpose is to cause the (instantiated) type of this
    // class to be registered with the Core/Disclosure/TypeDescription class when this
    // class' object code is originally loaded from the shared library.  The 'registerMe'
    // variable is not used for anything else in the program.
    static UnstructuredTypeDescription::Register registerMe;

  protected:
    UnstructuredSFCYVariable(const UnstructuredSFCYVariable<T>&);

  private:
    static UnstructuredTypeDescription* td;
    
    UnstructuredSFCYVariable<T>& operator=(const UnstructuredSFCYVariable<T>&);

    static Variable* maker();
  };

  template<class T>
  UnstructuredTypeDescription* UnstructuredSFCYVariable<T>::td = 0;

  // The following line is the initialization (creation) of the 'registerMe' static variable
  // (for each version of CCVariable (double, int, etc)).  Note, the 'registerMe' variable
  // is created when the object code is initially loaded (usually during intial program load
  // by the operating system).
  template<class T>
  UnstructuredTypeDescription::Register
  UnstructuredSFCYVariable<T>::registerMe( getUnstructuredTypeDescription() );
   
  template<class T>
  const UnstructuredTypeDescription*
  UnstructuredSFCYVariable<T>::getUnstructuredTypeDescription()
  {
    if(!td){
      td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::UnstructuredSFCYVariable,
                                  "UnstructuredSFCYVariable", &maker,
                                  fun_getUnstructuredTypeDescription((T*)0));
    }
    return td;
  }
    
  template<class T>
  Variable*
  UnstructuredSFCYVariable<T>::maker()
  {
    return scinew UnstructuredSFCYVariable<T>();
  }
   
  template<class T>
  UnstructuredSFCYVariable<T>::~UnstructuredSFCYVariable()
  {
  }
   
  template<class T>
  UnstructuredGridVariableBase*
  UnstructuredSFCYVariable<T>::clone()
  {
    return scinew UnstructuredSFCYVariable<T>(*this);
  }

  template<class T>
  const UnstructuredGridVariableBase*
  UnstructuredSFCYVariable<T>::clone() const
  {
    return scinew UnstructuredSFCYVariable<T>(*this);
  }

  template<class T>
  UnstructuredSFCYVariable<T>::UnstructuredSFCYVariable()
  {
  }

  template<class T>
  UnstructuredSFCYVariable<T>::UnstructuredSFCYVariable(const UnstructuredSFCYVariable<T>& copy)
    : UnstructuredGridVariable<T>(copy)
  {
  }
   
  template <class T>
  class constUnstructuredSFCYVariable : public constUnstructuredGridVariable<UnstructuredGridVariableBase, UnstructuredSFCYVariable<T>, T>
  {
  public:
    constUnstructuredSFCYVariable() : constUnstructuredGridVariable<UnstructuredGridVariableBase, UnstructuredSFCYVariable<T>, T>() {}
    constUnstructuredSFCYVariable(const UnstructuredSFCYVariable<T>& copy) : constUnstructuredGridVariable<UnstructuredGridVariableBase, UnstructuredSFCYVariable<T>, T>(copy) {}
  };

} // end namespace Uintah

#endif

