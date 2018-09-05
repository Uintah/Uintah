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


#ifndef UINTAH_HOMEBREW_UnstructuredParticleVariableBase_H
#define UINTAH_HOMEBREW_UnstructuredParticleVariableBase_H

#include <Core/Grid/Variables/UnstructuredParticleSubset.h>
#include <Core/Grid/Variables/UnstructuredVariable.h>
#include <Core/Grid/Variables/constUnstructuredVariable.h>

#include <Core/Util/Assert.h>

#include   <vector>


namespace Uintah {
  class BufferInfo;
  class OutputContext;
  class UnstructuredParticleSubset;
  class UnstructuredPatch;
  class ProcessorGroup;
  class UnstructuredTypeDescription;

/**************************************

CLASS
   UnstructuredParticleVariableBase
   
   Short description...

GENERAL INFORMATION

   UnstructuredParticleVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   UnstructuredParticleVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  typedef constUnstructuredVariableBase<UnstructuredParticleVariableBase> constUnstructuredParticleVariableBase;

   class UnstructuredParticleVariableBase : public UnstructuredVariable {
   public:
      
      virtual ~UnstructuredParticleVariableBase();

      //////////
      // Insert Documentation Here:
//      virtual const UnstructuredParticleVariableBase* clone() const = 0;
      virtual UnstructuredParticleVariableBase* clone() = 0;     
//      virtual const UnstructuredParticleVariableBase* cloneSubset(UnstructuredParticleSubset*) const = 0;
      virtual UnstructuredParticleVariableBase* cloneSubset(UnstructuredParticleSubset*) = 0;

      // Make a new default object of the base class.
      virtual UnstructuredParticleVariableBase* cloneType() const = 0;
      virtual constUnstructuredParticleVariableBase* cloneConstType() const = 0;

      // not something we normally do, but helps in AMR when we copy
      // data from one patch to another where they are the same boundaries
      // instead of copying all the data
      void setParticleSubset(UnstructuredParticleSubset* pset);

      virtual void copyData(const UnstructuredParticleVariableBase* src) = 0;
      
      virtual void allocate(const UnstructuredPatch*, const Uintah::IntVector& boundary) = 0; // will throw an InternalError
      virtual void allocate(UnstructuredParticleSubset*) = 0;
      virtual void allocate(int totalParticles) = 0;
      virtual void gather(UnstructuredParticleSubset* dest,
                          const std::vector<UnstructuredParticleSubset*> &subsets,
                          const std::vector<UnstructuredParticleVariableBase*> &srcs,
                          particleIndex extra = 0) = 0;
      virtual void gather(UnstructuredParticleSubset* dest,
                          const std::vector<UnstructuredParticleSubset*> &subsets,
                          const std::vector<UnstructuredParticleVariableBase*> &srcs,
                          const std::vector<const UnstructuredPatch*>& srcPatches,
                          particleIndex extra = 0) = 0;
      virtual void unpackMPI(void* buf, int bufsize, int* bufpos,
                             const ProcessorGroup* pg,
                             UnstructuredParticleSubset* pset) = 0;
      virtual void packMPI(void* buf, int bufsize, int* bufpos,
                           const ProcessorGroup* pg,
                           UnstructuredParticleSubset* pset) = 0;
      virtual void packMPI(void* buf, int bufsize, int* bufpos,
                           const ProcessorGroup* pg,
                           UnstructuredParticleSubset* pset, const UnstructuredPatch* forPatch) = 0;
      virtual void packsizeMPI(int* bufpos,
                               const ProcessorGroup* pg,
                               UnstructuredParticleSubset* pset) = 0;

      virtual int size() = 0;

      virtual size_t getDataSize() const = 0;

      virtual bool copyOut(void* dst) const = 0;

      //////////
      // Insert Documentation Here:
      UnstructuredParticleSubset* getParticleSubset() const {
        ASSERT(!isForeign());
         return d_pset;
      }

      virtual void* getBasePointer() const = 0;
      void getMPIBuffer(BufferInfo& buffer, UnstructuredParticleSubset* sendset);
      virtual const UnstructuredTypeDescription* virtualGetUnstructuredTypeDescription() const = 0;
     virtual RefCounted* getRefCounted() = 0;
     virtual void getSizeInfo(std::string& elems, unsigned long& totsize,
                              void*& ptr) const = 0;
   protected:
      UnstructuredParticleVariableBase(const UnstructuredParticleVariableBase&);
      UnstructuredParticleVariableBase(UnstructuredParticleSubset* pset);
      UnstructuredParticleVariableBase& operator=(const UnstructuredParticleVariableBase&);
      
      UnstructuredParticleSubset*  d_pset;

   private:
   };

} // End namespace Uintah

#endif
