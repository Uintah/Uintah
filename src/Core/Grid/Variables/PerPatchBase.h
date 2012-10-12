/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef UINTAH_HOMEBREW_PerPatchBase_H
#define UINTAH_HOMEBREW_PerPatchBase_H

#include <string>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Variables/Variable.h>

namespace Uintah {

  using SCIRun::IntVector;

  class Patch;
  class RefCounted;

/**************************************

CLASS
   PerPatchBase
   
   Short description...

GENERAL INFORMATION

   PerPatchBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   PerPatchBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  // inherits from Variable solely for the purpose of stuffing it in the DW
  class PerPatchBase : public Variable {
   public:
      
      virtual ~PerPatchBase();
      
      virtual const TypeDescription* virtualGetTypeDescription() const;
      virtual void copyPointer(Variable&) = 0;
      virtual PerPatchBase* clone() const = 0;
      virtual RefCounted* getRefCounted();
      virtual void getSizeInfo(std::string& elems, unsigned long& totsize,
                               void*& ptr) const = 0;

      // Only affects grid variables
      void offsetGrid(const SCIRun::IntVector& /*offset*/);
 
      virtual void emitNormal(std::ostream& out, const SCIRun::IntVector& l,
                              const SCIRun::IntVector& h, ProblemSpecP varnode, bool outputDoubleAsFloat );
      virtual void readNormal(std::istream& in, bool swapbytes);      
      virtual void allocate(const Patch* patch, const SCIRun::IntVector& boundary);

   protected:
      PerPatchBase(const PerPatchBase&);
      PerPatchBase();
      
   private:
      PerPatchBase& operator=(const PerPatchBase&);
   };
} // End namespace Uintah

#endif
