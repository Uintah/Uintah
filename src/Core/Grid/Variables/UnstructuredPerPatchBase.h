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


#ifndef UINTAH_HOMEBREW_UnstructuredPerPatchBase_H
#define UINTAH_HOMEBREW_UnstructuredPerPatchBase_H

#include <string>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Variables/UnstructuredVariable.h>

namespace Uintah {

  class UnstructuredPatch;
  class RefCounted;

/**************************************

CLASS
   UnstructuredPerPatchBase
   
   Short description...

GENERAL INFORMATION

   UnstructuredPerPatchBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   UnstructuredPerPatchBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  // inherits from UnstructuredVariable solely for the purpose of stuffing it in the DW
  class UnstructuredPerPatchBase : public UnstructuredVariable {
   public:
      
      virtual ~UnstructuredPerPatchBase();
      
      virtual const UnstructuredTypeDescription* virtualGetUnstructuredTypeDescription() const;
      virtual void copyPointer(UnstructuredVariable&) = 0;
      virtual UnstructuredPerPatchBase* clone() const = 0;
      virtual RefCounted* getRefCounted();
      virtual void getSizeInfo(std::string& elems, unsigned long& totsize,
                               void*& ptr) const = 0;

      virtual size_t getDataSize() const = 0;
      virtual bool copyOut(void* dst) const = 0;
      virtual void* getBasePointer() const = 0;

      // Only affects grid variables
      void offsetGrid(const IntVector& /*offset*/);
 
      virtual void emitNormal(std::ostream& out, const IntVector& l,
                              const IntVector& h, ProblemSpecP varnode, bool outputDoubleAsFloat );
      virtual void readNormal(std::istream& in, bool swapbytes);      
      virtual void allocate(const UnstructuredPatch* patch, const IntVector& boundary);

   protected:
      UnstructuredPerPatchBase(const UnstructuredPerPatchBase&);
      UnstructuredPerPatchBase();
      
   private:
      UnstructuredPerPatchBase& operator=(const UnstructuredPerPatchBase&);
   };
} // End namespace Uintah

#endif
