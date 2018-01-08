#ifndef UINTAH_HOMEBREW_SOLEVARIABLEBASE_H
#define UINTAH_HOMEBREW_SOLEVARIABLEBASE_H

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

#include <Core/Grid/Variables/Variable.h>
#include <Core/Parallel/UintahMPI.h>

#include <iosfwd>

namespace Uintah {

/**************************************

CLASS
   SoleVariableBase
   
   Short description...

GENERAL INFORMATION

   SoleVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   SoleVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SoleVariableBase : public Variable {

   public:
      
      virtual ~SoleVariableBase();

      virtual void copyPointer(Variable&) = 0;
      virtual SoleVariableBase* clone() const = 0; 
      virtual const TypeDescription* virtualGetTypeDescription() const;
      virtual RefCounted* getRefCounted();
      virtual void getSizeInfo(std::string& elems,unsigned long& totsize,
                               void*& ptr) const = 0;
      virtual size_t getDataSize() const = 0;
      virtual bool copyOut(void* dst) const = 0;
      virtual void emitNormal(std::ostream& out, const IntVector& l,
                              const IntVector& h, ProblemSpecP varnode, 
                              bool outputDoubleAsFloat );
      virtual void readNormal(std::istream& in, bool swapbytes);      
      virtual void allocate(const Patch* patch, const IntVector& boundary);

   protected:
      SoleVariableBase(const SoleVariableBase&);
      SoleVariableBase();
      
   private:
      SoleVariableBase& operator=(const SoleVariableBase&);
   };
} // End namespace Uintah

#endif // UINTAH_HOMEBREW_SOLEVARIABLEBASE_H
