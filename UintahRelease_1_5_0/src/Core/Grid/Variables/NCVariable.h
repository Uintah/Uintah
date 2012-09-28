/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_NCVARIABLE_H
#define UINTAH_HOMEBREW_NCVARIABLE_H

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
   NCVariable
   
   Short description...

GENERAL INFORMATION

   NCVariable.h

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
  class NCVariable : public GridVariable<T> {
    friend class constVariable<GridVariableBase, NCVariable<T>, T, const IntVector&>;
  public:
    NCVariable();
    virtual ~NCVariable();
      
    //////////
    // Insert Documentation Here:
    const TypeDescription* virtualGetTypeDescription() const 
    { return getTypeDescription(); }
    static const TypeDescription* getTypeDescription();
    
    virtual GridVariableBase* clone();
    virtual const GridVariableBase* clone() const;
    virtual GridVariableBase* cloneType() const
    { return scinew NCVariable<T>(); }
    
    // allocate(IntVector, IntVector) is hidden without this
    using GridVariable<T>::allocate;
    virtual void allocate(const Patch* patch, const IntVector& boundary)
    {      
      IntVector l,h;
      patch->computeVariableExtents(Patch::NodeBased, boundary, 
                                    Ghost::None, 0, l, h);
      GridVariable<T>::allocate(l, h);
    }

    static TypeDescription::Register registerMe;

  protected:
    NCVariable(const NCVariable<T>&);

  private:
    static TypeDescription* td;

    NCVariable<T>& operator=(const NCVariable<T>&);

    static Variable* maker();
  };
  
  template<class T>
  TypeDescription* NCVariable<T>::td = 0;

  template<class T>
  TypeDescription::Register
  NCVariable<T>::registerMe(getTypeDescription());
   
  template<class T>
  const TypeDescription*
  NCVariable<T>::getTypeDescription()
  {
    if(!td){
      td = scinew TypeDescription(TypeDescription::NCVariable,
                                  "NCVariable", &maker,
                                  fun_getTypeDescription((T*)0));
    }
    return td;
  }
    
  template<class T>
  Variable*
  NCVariable<T>::maker()
  {
    return scinew NCVariable<T>();
  }
   
  template<class T>
  NCVariable<T>::~NCVariable()
  {
  }
   
  template<class T>
  GridVariableBase*
  NCVariable<T>::clone()
  {
    return scinew NCVariable<T>(*this);
  }

  template<class T>
  const GridVariableBase*
  NCVariable<T>::clone() const
  {
    return scinew NCVariable<T>(*this);
  }

  template<class T>
  NCVariable<T>::NCVariable()
  {
  }

  template<class T>
  NCVariable<T>::NCVariable(const NCVariable<T>& copy)
    : GridVariable<T>(copy)
  {
  }

  template <class T>
  class constNCVariable : public constGridVariable<GridVariableBase, NCVariable<T>, T>
  {
  public:
    constNCVariable() : constGridVariable<GridVariableBase, NCVariable<T>, T>() {}
    constNCVariable(const NCVariable<T>& copy) : constGridVariable<GridVariableBase, NCVariable<T>, T>(copy) {}
  };


} // end namespace Uintah

#endif

