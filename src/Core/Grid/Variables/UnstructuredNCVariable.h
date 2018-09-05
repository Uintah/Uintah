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

#ifndef UINTAH_UNSTRUCTURED_NCVARIABLE_H
#define UINTAH_UNSTRUCTURED_NCVARIABLE_H

#include <Core/Grid/Variables/UnstructuredGridVariable.h>
#include <Core/Grid/Variables/constUnstructuredGridVariable.h>
#include <Core/Grid/UnstructuredPatch.h>

namespace Uintah {

  class UnstructuredTypeDescription;

  /**************************************

CLASS
   UnstructuredNCVariable
   
   Short description...

GENERAL INFORMATION

   UnstructuredNCVariable.h

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
  class UnstructuredNCVariable : public UnstructuredGridVariable<T> {
    friend class constVariable<UnstructuredGridVariableBase, UnstructuredNCVariable<T>, T, const IntVector&>;
  public:
    UnstructuredNCVariable();
    virtual ~UnstructuredNCVariable();
      
    //////////
    // Insert Documentation Here:
    const UnstructuredTypeDescription* virtualGetUnstructuredTypeDescription() const 
    { return getUnstructuredTypeDescription(); }
    static const UnstructuredTypeDescription* getUnstructuredTypeDescription();
    
    virtual UnstructuredGridVariableBase* clone();
    virtual const UnstructuredGridVariableBase* clone() const;
    virtual UnstructuredGridVariableBase* cloneType() const
    { return scinew UnstructuredNCVariable<T>(); }
    
    // allocate(IntVector, IntVector) is hidden without this
    using UnstructuredGridVariable<T>::allocate;
    virtual void allocate(const UnstructuredPatch* patch, const IntVector& boundary)
    {      
      IntVector l,h;
      patch->computeVariableExtents(UnstructuredPatch::NodeBased, boundary, 
                                    Ghost::None, 0, l, h);
      UnstructuredGridVariable<T>::allocate(l, h);
    }

    // Static variable whose entire purpose is to cause the (instantiated) type of this
    // class to be registered with the Core/Disclosure/TypeDescription class when this
    // class' object code is originally loaded from the shared library.  The 'registerMe'
    // variable is not used for anything else in the program.
    static UnstructuredTypeDescription::Register registerMe;

  protected:
    UnstructuredNCVariable(const UnstructuredNCVariable<T>&);

  private:
    static UnstructuredTypeDescription* td;

    UnstructuredNCVariable<T>& operator=(const UnstructuredNCVariable<T>&);

    static Variable* maker();
  };
  
  template<class T>
  UnstructuredTypeDescription* UnstructuredNCVariable<T>::td = 0;

  // The following line is the initialization (creation) of the 'registerMe' static variable
  // (for each version of CCVariable (double, int, etc)).  Note, the 'registerMe' variable
  // is created when the object code is initially loaded (usually during intial program load
  // by the operating system).
  template<class T>
  UnstructuredTypeDescription::Register
  UnstructuredNCVariable<T>::registerMe( getUnstructuredTypeDescription() );

  template<class T>
  const UnstructuredTypeDescription*
  UnstructuredNCVariable<T>::getUnstructuredTypeDescription()
  {
    if(!td){
      td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::UnstructuredNCVariable,
                                  "UnstructuredNCVariable", &maker,
                                  fun_getUnstructuredTypeDescription((T*)0));
    }
    return td;
  }
    
  template<class T>
  Variable*
  UnstructuredNCVariable<T>::maker()
  {
    return scinew UnstructuredNCVariable<T>();
  }
   
  template<class T>
  UnstructuredNCVariable<T>::~UnstructuredNCVariable()
  {
  }
   
  template<class T>
  UnstructuredGridVariableBase*
  UnstructuredNCVariable<T>::clone()
  {
    return scinew UnstructuredNCVariable<T>(*this);
  }

  template<class T>
  const UnstructuredGridVariableBase*
  UnstructuredNCVariable<T>::clone() const
  {
    return scinew UnstructuredNCVariable<T>(*this);
  }

  template<class T>
  UnstructuredNCVariable<T>::UnstructuredNCVariable()
  {
  }

  template<class T>
  UnstructuredNCVariable<T>::UnstructuredNCVariable(const UnstructuredNCVariable<T>& copy)
    : UnstructuredGridVariable<T>(copy)
  {
  }

  template <class T>
  class constUnstructuredNCVariable : public constUnstructuredGridVariable<UnstructuredGridVariableBase, UnstructuredNCVariable<T>, T>
  {
  public:
    constUnstructuredNCVariable() : constUnstructuredGridVariable<UnstructuredGridVariableBase, UnstructuredNCVariable<T>, T>() {}
    constUnstructuredNCVariable(const UnstructuredNCVariable<T>& copy) : constUnstructuredGridVariable<UnstructuredGridVariableBase, UnstructuredNCVariable<T>, T>(copy) {}
  };


} // end namespace Uintah

#endif

