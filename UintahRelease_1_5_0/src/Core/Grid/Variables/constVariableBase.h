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
   
    virtual const VariableBase& getBaseRep() const = 0;
    
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

