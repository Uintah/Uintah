/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
#include <Core/Malloc/AllocatorTags.hpp>

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

//  template < typename U >
//  using allocator_type = TrackingAllocator<   U
//                                            , Uintah::Tags::CCVariable
//                                            , Uintah::MallocAllocator
//                                          >;

  friend class constVariable<GridVariableBase, CCVariable<T>, T, const IntVector&>;

  public:
    CCVariable() = default;

    CCVariable& operator=(const CCVariable&) = delete;

    virtual ~CCVariable() {}
      
    //////////
    // Insert Documentation Here:
    const TypeDescription* virtualGetTypeDescription() const 
    { return getTypeDescription(); }

    static const TypeDescription* getTypeDescription()
    {
      if (!td) {
        td = new TypeDescription(TypeDescription::CCVariable, "CCVariable", &maker, fun_getTypeDescription((T*)0));
      }
      return td;
    }
    
    virtual GridVariableBase* clone()
    {
      return new CCVariable<T>(*this);
    }

    virtual const GridVariableBase* clone() const
    {
      return new CCVariable<T>(*this);
    }

    virtual GridVariableBase* cloneType() const
    {
      return new CCVariable<T>();
    }
    
    // allocate(IntVector, IntVector) is hidden without this
    using GridVariable<T>::allocate;

    virtual void allocate(const Patch* patch, const IntVector& boundary)
    {      
      IntVector l,h;
      patch->computeVariableExtents(Patch::CellBased, boundary, Ghost::None, 0, l, h);
      GridVariable<T>::allocate(l, h);
    }

  private:
    static Variable* maker()
    {
      return new CCVariable<T>();
    }

  protected:
    CCVariable(const CCVariable&) = default;

  public: // public data member
    static TypeDescription::Register registerMe;

  private: // private data members
    static TypeDescription* td;

};

  // declare linkage
  template<class T> TypeDescription* CCVariable<T>::td = 0;
  template<class T> TypeDescription::Register CCVariable<T>::registerMe(getTypeDescription());



  // constCCVariable class
  template <class T>
  class constCCVariable : public constGridVariable<GridVariableBase, CCVariable<T>, T>
  {
    public:
      constCCVariable() : constGridVariable<GridVariableBase, CCVariable<T>, T>() {}
      constCCVariable(const CCVariable<T>& copy) : constGridVariable<GridVariableBase, CCVariable<T>, T>(copy) {}
  };


} // end namespace Uintah

#endif

