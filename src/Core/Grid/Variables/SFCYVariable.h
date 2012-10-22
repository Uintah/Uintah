/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_HOMEBREW_SFCYVARIABLE_H
#define UINTAH_HOMEBREW_SFCYVARIABLE_H

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
   SFCYVariable
   
   Short description...

GENERAL INFORMATION

   SFCYVariable.h

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
  class SFCYVariable : public GridVariable<T> {
    friend class constVariable<GridVariableBase, SFCYVariable<T>, T, const IntVector&>;
  public:
    SFCYVariable();
    virtual ~SFCYVariable();
      
    //////////
    // Insert Documentation Here:
    const TypeDescription* virtualGetTypeDescription() const 
    { return getTypeDescription(); }
    static const TypeDescription* getTypeDescription();
    

    virtual GridVariableBase* clone();
    virtual const GridVariableBase* clone() const;
    virtual GridVariableBase* cloneType() const
    { return scinew SFCYVariable<T>(); }
    
    // allocate(IntVector, IntVector) is hidden without this
    using GridVariable<T>::allocate;
    virtual void allocate(const Patch* patch, const IntVector& boundary)
    {      
      IntVector l,h;
      patch->computeVariableExtents(Patch::YFaceBased, boundary, 
                                    Ghost::None, 0, l, h);
      this->allocate(l, h);
    }

    static TypeDescription::Register registerMe;

  protected:
    SFCYVariable(const SFCYVariable<T>&);

  private:
    static TypeDescription* td;
    
    SFCYVariable<T>& operator=(const SFCYVariable<T>&);

    static Variable* maker();
  };

  template<class T>
  TypeDescription* SFCYVariable<T>::td = 0;

  template<class T>
  TypeDescription::Register
  SFCYVariable<T>::registerMe(getTypeDescription());
   
  template<class T>
  const TypeDescription*
  SFCYVariable<T>::getTypeDescription()
  {
    if(!td){
      td = scinew TypeDescription(TypeDescription::SFCYVariable,
                                  "SFCYVariable", &maker,
                                  fun_getTypeDescription((T*)0));
    }
    return td;
  }
    
  template<class T>
  Variable*
  SFCYVariable<T>::maker()
  {
    return scinew SFCYVariable<T>();
  }
   
  template<class T>
  SFCYVariable<T>::~SFCYVariable()
  {
  }
   
  template<class T>
  GridVariableBase*
  SFCYVariable<T>::clone()
  {
    return scinew SFCYVariable<T>(*this);
  }

  template<class T>
  const GridVariableBase*
  SFCYVariable<T>::clone() const
  {
    return scinew SFCYVariable<T>(*this);
  }

  template<class T>
  SFCYVariable<T>::SFCYVariable()
  {
  }

  template<class T>
  SFCYVariable<T>::SFCYVariable(const SFCYVariable<T>& copy)
    : GridVariable<T>(copy)
  {
  }
   
  template <class T>
  class constSFCYVariable : public constGridVariable<GridVariableBase, SFCYVariable<T>, T>
  {
  public:
    constSFCYVariable() : constGridVariable<GridVariableBase, SFCYVariable<T>, T>() {}
    constSFCYVariable(const SFCYVariable<T>& copy) : constGridVariable<GridVariableBase, SFCYVariable<T>, T>(copy) {}
  };

} // end namespace Uintah

#endif

