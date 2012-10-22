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

#ifndef UINTAH_HOMEBREW_PARTICLEVARIABLE_H
#define UINTAH_HOMEBREW_PARTICLEVARIABLE_H


#include <TauProfilerForSCIRun.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Endian.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Variables/ParticleVariableBase.h>
#include <Core/Grid/Variables/constGridVariable.h>
#include <CCA/Ports/InputContext.h>
#include <CCA/Ports/OutputContext.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Grid/Variables/ParticleData.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Patch.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/IO/SpecializedRunLengthEncoder.h>

#ifndef _WIN32
#include <unistd.h>
#endif
#include <iostream>


namespace Uintah {

  using SCIRun::InternalError;

  class ProcessorGroup;
class TypeDescription;

/**************************************

CLASS
ParticleVariable
  
Short description...

GENERAL INFORMATION

ParticleVariable.h

Steven G. Parker
Department of Computer Science
University of Utah

Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
Particle_Variable

DESCRIPTION
Long description...
  
WARNING
  
****************************************/

template<class T>
class ParticleVariable : public ParticleVariableBase {
  friend class constVariable<ParticleVariableBase, ParticleVariable<T>, T, particleIndex>;
public:
  ParticleVariable();
  virtual ~ParticleVariable();
  ParticleVariable(ParticleSubset* pset);
  ParticleVariable(ParticleData<T>*, ParticleSubset* pset);
      
    //////////
    // Insert Documentation Here:
  static const TypeDescription* getTypeDescription();

  //////////
  // Insert Documentation Here:
  void resync() {
    d_pdata->resize(getParticleSubset()->numParticles());
  }
      
  //////////
  // Insert Documentation Here:
  virtual ParticleVariableBase* clone();
  virtual const ParticleVariableBase* clone() const;
  virtual ParticleVariableBase* cloneSubset(ParticleSubset*);
  virtual const ParticleVariableBase* cloneSubset(ParticleSubset*) const;

  virtual ParticleVariableBase* cloneType() const
  { return scinew ParticleVariable<T>(); }
  virtual constParticleVariableBase* cloneConstType() const
  { return scinew constVariable<ParticleVariableBase, ParticleVariable<T>, T, particleIndex>();
  }

  
  void copyData(const ParticleVariable<T>& src);
  virtual void copyData(const ParticleVariableBase* src)
  { copyData(castFromBase(src)); }
  
  //////////
  // Insert Documentation Here:
  inline T& operator[](particleIndex idx) {
    ASSERTRANGE(idx, 0, (particleIndex)d_pdata->size);
    return d_pdata->data[idx];
  }
      
  //////////
  // Insert Documentation Here:
  inline const T& operator[](particleIndex idx) const {
    ASSERTRANGE(idx, 0, (particleIndex)d_pdata->size);
    return d_pdata->data[idx];
  }

  virtual void copyPointer(ParticleVariable<T>&);
  virtual void copyPointer(Variable&);
  virtual void allocate(ParticleSubset*);
  virtual void allocate(int totalParticles);
  virtual void allocate(const Patch*, const IntVector& /*boundary*/)
  { SCI_THROW(InternalError("Should not call ParticleVariable<T>::allocate(const Patch*), use allocate(ParticleSubset*) instead.", __FILE__, __LINE__)); }

  virtual int size() { return d_pdata->size; }

  // specialized for T=Point
  virtual void gather(ParticleSubset* dest,
                      const std::vector<ParticleSubset*> &subsets,
                      const std::vector<ParticleVariableBase*> &srcs,
                      const std::vector<const Patch*>& /*srcPatches*/,
                      particleIndex extra = 0);  
  virtual void gather(ParticleSubset* dest,
                      const std::vector<ParticleSubset*> &subsets,
                      const std::vector<ParticleVariableBase*> &srcs,
                      particleIndex extra = 0);
  
  virtual void unpackMPI(void* buf, int bufsize, int* bufpos,
                         const ProcessorGroup* pg, ParticleSubset* pset);
  virtual void packMPI(void* buf, int bufsize, int* bufpos,
                       const ProcessorGroup* pg, ParticleSubset* pset);
  // specialized for T=Point
  virtual void packMPI(void* buf, int bufsize, int* bufpos,
                       const ProcessorGroup* pg, ParticleSubset* pset,
                       const Patch* /*forPatch*/);
  virtual void packsizeMPI(int* bufpos,
                           const ProcessorGroup* pg,
                           ParticleSubset* pset);
  virtual void emitNormal(ostream& out, const IntVector& l,
                          const IntVector& h, ProblemSpecP varnode, bool outputDoubleAsFloat);
  virtual bool emitRLE(ostream& out, const IntVector& l, const IntVector& h,
                       ProblemSpecP varnode);
  
  virtual void readNormal(istream& in, bool swapBytes);
  virtual void readRLE(istream& in, bool swapBytes, int nByteMode);
  
  virtual void* getBasePointer() const;
  virtual const TypeDescription* virtualGetTypeDescription() const;
  virtual RefCounted* getRefCounted() {
    return d_pdata;
  }
  virtual void getSizeInfo(std::string& elems, unsigned long& totsize,
                           void*& ptr) const {
    std::ostringstream str;
    str << getParticleSubset()->numParticles();
    elems=str.str();
    totsize = getParticleSubset()->numParticles()*sizeof(T);
    ptr = getBasePointer();
  }
protected:
  static TypeDescription* td;
  ParticleVariable(const ParticleVariable<T>&);
  ParticleVariable<T>& operator=(const ParticleVariable<T>&);

private:
    //////////
    // Insert Documentation Here:
  ParticleData<T>* d_pdata;
  Vector offset_; // only used when T is Point

  static const ParticleVariable<T>& castFromBase(const ParticleVariableBase* srcptr);
  static TypeDescription::Register registerMe;
  static Variable* maker();
};

  template<class T>
  TypeDescription* ParticleVariable<T>::td = 0;

  template<class T>
  TypeDescription::Register ParticleVariable<T>::registerMe(getTypeDescription());

  template<class T>
  const TypeDescription*
  ParticleVariable<T>::getTypeDescription()
  {
    if(!td){
      td = scinew TypeDescription(TypeDescription::ParticleVariable,
                                  "ParticleVariable", &maker,
                                  fun_getTypeDescription((T*)0));
    }
    return td;
  }
   
  template<class T>
  Variable*
  ParticleVariable<T>::maker()
  {
    return scinew ParticleVariable<T>();
  }
   
  template<class T>
  ParticleVariable<T>::ParticleVariable()
    : ParticleVariableBase(0), d_pdata(0)
  {
  }
   
  template<class T>
  ParticleVariable<T>::~ParticleVariable()
  {
    if(d_pdata && d_pdata->removeReference())
      delete d_pdata;
  }
   
  template<class T>
  ParticleVariable<T>::ParticleVariable(ParticleSubset* pset)
    : ParticleVariableBase(pset)
  {
    d_pdata=scinew ParticleData<T>(pset->numParticles());
    d_pdata->addReference();
  }
   
  template<class T>
  void ParticleVariable<T>::allocate(int totalParticles)
  {
    ASSERT(isForeign());
    ASSERT(d_pset == 0);

    // this is a pset-less storage as it could have several.  Should be used for
    // foreign data only.  To iterate over particles in this pset, use gather
    d_pdata=scinew ParticleData<T>(totalParticles);
    d_pdata->addReference();
  }

  template<class T>
  void ParticleVariable<T>::allocate(ParticleSubset* pset)
  {
    TAU_PROFILE_TIMER(t1, "Release old ParticleVariable<T>::allocate()", "", TAU_USER3);
    TAU_PROFILE_TIMER(t2, "Allocate Data ParticleVariable<T>::allocate()", "", TAU_USER3);

    TAU_PROFILE_START(t1);
    if(d_pdata && d_pdata->removeReference())
      delete d_pdata;
    if(d_pset && d_pset->removeReference())
      delete d_pset;
    TAU_PROFILE_STOP(t1);

    d_pset=pset;
    d_pset->addReference();

    TAU_PROFILE_START(t2);
    d_pdata=scinew ParticleData<T>(pset->numParticles());
    TAU_PROFILE_STOP(t2);

    d_pdata->addReference();
  }
   
  template<class T>
  ParticleVariableBase*
  ParticleVariable<T>::clone()
  { return scinew ParticleVariable<T>(*this); }

  template<class T>
  const ParticleVariableBase*
  ParticleVariable<T>::clone() const
  { return scinew ParticleVariable<T>(*this); }
   
  template<class T>
  ParticleVariableBase*
  ParticleVariable<T>::cloneSubset(ParticleSubset* pset)
  { return scinew ParticleVariable<T>(d_pdata, pset); }

  template<class T>
  const ParticleVariableBase*
  ParticleVariable<T>::cloneSubset(ParticleSubset* pset) const
  { return scinew ParticleVariable<T>(d_pdata, pset); }

  template<class T>
  const ParticleVariable<T>& ParticleVariable<T>::castFromBase(const ParticleVariableBase* srcptr)
  {
    const ParticleVariable<T>* c = dynamic_cast<const ParticleVariable<T>* >(srcptr);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in Particle variable", __FILE__, __LINE__));
    return *c;
  }

  template<class T>
    void ParticleVariable<T>::copyData(const ParticleVariable<T>& src)
  {
    ASSERT(*d_pset == *src.d_pset);
    *d_pdata = *src.d_pdata;
  }


  template<class T>
  ParticleVariable<T>::ParticleVariable(ParticleData<T>* pdata,
                                        ParticleSubset* pset)
    : ParticleVariableBase(pset), d_pdata(pdata)
  {
    if(d_pdata)
      d_pdata->addReference();
  }
   
  template<class T>
  ParticleVariable<T>::ParticleVariable(const ParticleVariable<T>& copy)
    : ParticleVariableBase(copy), d_pdata(copy.d_pdata)
  {
    if(d_pdata)
      d_pdata->addReference();
  }
   
  template<class T>
  void
  ParticleVariable<T>::copyPointer(ParticleVariable<T>& copy)
  {
    if(this != &copy){
      ParticleVariableBase::operator=(copy);
      if(d_pdata && d_pdata->removeReference())
        delete d_pdata;
      d_pdata = copy.d_pdata;
      if(d_pdata)
        d_pdata->addReference();
    }
  }
   
  template<class T>
  void
  ParticleVariable<T>::copyPointer(Variable& copy)
  {
    ParticleVariable<T>* c = dynamic_cast<ParticleVariable<T>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in particle variable", __FILE__, __LINE__));
    copyPointer(*c);
  }
  
  // specialization for T=Point
  template <>
   void ParticleVariable<Point>::gather(ParticleSubset* pset,
                                       const vector<ParticleSubset*> &subsets,
                                       const vector<ParticleVariableBase*> &srcs,
                                       const vector<const Patch*>& srcPatches,
                                       particleIndex extra);

  template<class T>
    void ParticleVariable<T>::gather(ParticleSubset* pset,
                                     const vector<ParticleSubset*> &subsets,
                                     const vector<ParticleVariableBase*> &srcs,
                                     const vector<const Patch*>& /*srcPatches*/,
                                     particleIndex extra)
  { gather(pset, subsets, srcs, extra); }

template<class T>
  void
  ParticleVariable<T>::gather(ParticleSubset* pset,
                              const std::vector<ParticleSubset*> &subsets,
                              const std::vector<ParticleVariableBase*> &srcs,
                              particleIndex extra)
  {
    if(d_pdata && d_pdata->removeReference())
      delete d_pdata;
    if(d_pset && d_pset->removeReference())
      delete d_pset;
    d_pset = pset;
    pset->addReference();
    d_pdata=scinew ParticleData<T>(pset->numParticles());
    d_pdata->addReference();
    ASSERTEQ(subsets.size(), srcs.size());
    ParticleSubset::iterator dstiter = pset->begin();
    for(int i=0;i<(int)subsets.size();i++){
      ParticleVariable<T>* srcptr = dynamic_cast<ParticleVariable<T>*>(srcs[i]);
      if(!srcptr)
        SCI_THROW(TypeMismatchException("Type mismatch in ParticleVariable::gather", __FILE__, __LINE__));
      ParticleVariable<T>& src = *srcptr;
      ParticleSubset* subset = subsets[i];
      for(ParticleSubset::iterator srciter = subset->begin();
          srciter != subset->end(); srciter++){
        (*this)[*dstiter] = src[*srciter];
        dstiter++;
      }
    }
    ASSERT(dstiter+extra == pset->end());
    extra = extra;   // This is to shut up the REMARKS from the MIPS compiler
  }
  
  template<class T>
  void*
  ParticleVariable<T>::getBasePointer() const
  {
    return &d_pdata->data[0];
  }
  
  template<class T>
  const TypeDescription*
  ParticleVariable<T>::virtualGetTypeDescription() const
  {
    return getTypeDescription();
  }
  
  template<class T>
  void
  ParticleVariable<T>::unpackMPI(void* buf, int bufsize, int* bufpos,
                                 const ProcessorGroup* pg,
                                 ParticleSubset* pset)
  {
    // This should be fixed for variable sized types!
    const TypeDescription* td = getTypeDescription()->getSubType();
    if(td->isFlat()){
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        MPI_Unpack(buf, bufsize, bufpos,
                   &d_pdata->data[*iter], 1, td->getMPIType(),
                   pg->getComm());
      }
    } else {
      SCI_THROW(InternalError("packMPI not finished\n", __FILE__, __LINE__));
    }
  }
  
  // specialized for T=Point
  template<>
   void
    ParticleVariable<Point>::packMPI(void* buf, int bufsize, int* bufpos,
                                     const ProcessorGroup* pg,
                                     ParticleSubset* pset,
                                     const Patch* forPatch);
  template<class T>
  void
    ParticleVariable<T>::packMPI(void* buf, int bufsize, int* bufpos,
                                 const ProcessorGroup* pg,
                                 ParticleSubset* pset,
                                 const Patch* /*forPatch*/)
    { packMPI(buf, bufsize, bufpos, pg, pset); }


  template<class T>
  void
  ParticleVariable<T>::packMPI(void* buf, int bufsize, int* bufpos,
                               const ProcessorGroup* pg,
                               ParticleSubset* pset)
  {
    // This should be fixed for variable sized types!
    const TypeDescription* td = getTypeDescription()->getSubType();
    if(td->isFlat()){
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        MPI_Pack(&d_pdata->data[*iter], 1, td->getMPIType(),
                 buf, bufsize, bufpos, pg->getComm());
      }
    } else {
      SCI_THROW(InternalError("packMPI not finished\n", __FILE__, __LINE__));
    }
  }

  template<class T>
  void
  ParticleVariable<T>::packsizeMPI(int* bufpos,
                                   const ProcessorGroup* pg,
                                   ParticleSubset* pset)
  {
    // This should be fixed for variable sized types!
    const TypeDescription* td = getTypeDescription()->getSubType();
    int n = pset->numParticles();
    if(td->isFlat()){
      int size;
      MPI_Pack_size(n, td->getMPIType(), pg->getComm(), &size);
      (*bufpos)+= size;
    } else {
      SCI_THROW(InternalError("packsizeMPI not finished\n", __FILE__, __LINE__));
    }
  }

  // Specialized in ParticleVariable_special.cc
  template<>
   void
  ParticleVariable<double>::emitNormal(ostream& out, const IntVector&,
                                  const IntVector&, ProblemSpecP varnode, bool outputDoubleAsFloat );

  template<class T>
  void
  ParticleVariable<T>::emitNormal(ostream& out, const IntVector&,
                                  const IntVector&, ProblemSpecP varnode, bool /*outputDoubleAsFloat*/ )
  {
    const TypeDescription* td = fun_getTypeDescription((T*)0);

    if (varnode->findBlock("numParticles") == 0) {
      varnode->appendElement("numParticles", d_pset->numParticles());
    }
    if(!td->isFlat()){
      SCI_THROW(InternalError("Cannot yet write non-flat objects!\n", __FILE__, __LINE__));
    }
    else {
      // This could be optimized...
      ParticleSubset::iterator iter = d_pset->begin();
      while(iter != d_pset->end()){
        particleIndex start = *iter;
        iter++;
        particleIndex end = start+1;
        while(iter != d_pset->end() && *iter == end) {
          end++;
          iter++;
        }
        ssize_t size = (ssize_t)(sizeof(T)*(end-start));
        out.write((char*)&(*this)[start], size);
      }
    }
  }

  template<class T>
  bool
  ParticleVariable<T>::emitRLE(ostream& out, const IntVector& /*l*/,
                               const IntVector& /*h*/, ProblemSpecP varnode)
  {
    const TypeDescription* td = fun_getTypeDescription((T*)0);
    if (varnode->findBlock("numParticles") == 0) {
      varnode->appendElement("numParticles", d_pset->numParticles());
    }
    if(!td->isFlat()){
      SCI_THROW(InternalError("Cannot yet write non-flat objects!\n", __FILE__, __LINE__));
    }
    else {
      // emit in runlength encoded format
      SCIRun::RunLengthEncoder<T> rle;
      ParticleSubset::iterator iter = d_pset->begin();
      for ( ; iter != d_pset->end(); iter++)
        rle.addItem((*this)[*iter]);
      rle.write(out);
    }
    return true;
  }
  
  template<class T>
  void
  ParticleVariable<T>::readNormal(istream& in, bool swapBytes)
  {
    const TypeDescription* td = fun_getTypeDescription((T*)0);
    if(!td->isFlat()) {
      SCI_THROW(InternalError("Cannot yet read non-flat objects!\n", __FILE__, __LINE__));
    }
    else {
      // This could be optimized...
      ParticleSubset::iterator iter = d_pset->begin();
      while(iter != d_pset->end()){
        particleIndex start = *iter;
        iter++;
        particleIndex end = start+1;
        while(iter != d_pset->end() && *iter == end) {
          end++;
          iter++;
        }
        ssize_t size = (ssize_t)(sizeof(T)*(end-start));
        in.read((char*)&(*this)[start], size);
        if (swapBytes) {
          for (particleIndex idx = start; idx != end; idx++) {
            SCIRun::swapbytes((*this)[idx]);
          }
        }
      }
    }
  }

  template<class T>
  void
  ParticleVariable<T>::readRLE(istream& in, bool swapBytes, int nByteMode)
  {
    const TypeDescription* td = fun_getTypeDescription((T*)0);
    if(!td->isFlat()) {
      SCI_THROW(InternalError("Cannot yet read non-flat objects!\n", __FILE__, __LINE__));
    }
    else {
      SCIRun::RunLengthEncoder<T> rle;
      rle.read(in, swapBytes, nByteMode);
      ParticleSubset::iterator iter = d_pset->begin();
      typename SCIRun::RunLengthEncoder<T>::iterator rle_iter = rle.begin();
      for ( ; iter != d_pset->end() && rle_iter != rle.end();
            iter++, rle_iter++)
        (*this)[*iter] = *rle_iter;

      if ((rle_iter != rle.end()) || (iter != d_pset->end()))
        SCI_THROW(InternalError("ParticleVariable::read RLE data is not consistent with the particle subset size", __FILE__, __LINE__));
    }
  }

  template <class T>
  class constParticleVariable : public constVariable<ParticleVariableBase, ParticleVariable<T>, T, particleIndex>
  {
  public:
    constParticleVariable()
      : constVariable<ParticleVariableBase, ParticleVariable<T>, T, particleIndex>() {}
    
    constParticleVariable(const ParticleVariable<T>& copy)
      : constVariable<ParticleVariableBase, ParticleVariable<T>, T, particleIndex>(copy) {}

    ParticleSubset* getParticleSubset() const {
      return this->rep_.getParticleSubset();
    }
  };

} // End namespace Uintah

#ifdef __PGI
#include <Core/Grid/Variables/ParticleVariable_special.cc>
#endif

#endif
