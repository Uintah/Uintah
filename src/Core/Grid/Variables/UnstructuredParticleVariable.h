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

#ifndef UINTAH_HOMEBREW_UNSTRUCTURED_PARTICLEVARIABLE_H
#define UINTAH_HOMEBREW_UNSTRUCTURED_PARTICLEVARIABLE_H


#include <Core/Util/FancyAssert.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Endian.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Variables/UnstructuredParticleVariableBase.h>
#include <Core/Grid/Variables/UnstructuredVariable.h>
#include <Core/Grid/Variables/constUnstructuredVariable.h>
#include <CCA/Ports/InputContext.h>
#include <CCA/Ports/OutputContext.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Grid/Variables/UnstructuredParticleData.h>
#include <Core/Grid/Variables/UnstructuredParticleSubset.h>
#include <Core/Grid/UnstructuredPatch.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Disclosure/UnstructuredTypeDescription.h>
#include <Core/Disclosure/UnstructuredTypeUtils.h>
#include <Core/IO/SpecializedRunLengthEncoder.h>
#include <iostream>
#include <cstring>


namespace Uintah {

  class ProcessorGroup;
  class UnstructuredTypeDescription;

/**************************************

CLASS
UnstructuredParticleVariable
  
Short description...

GENERAL INFORMATION

UnstructuredParticleVariable.h

Steven G. Parker
Department of Computer Science
University of Utah

Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
UnstructuredParticle_Variable

DESCRIPTION
Long description...
  
WARNING
  
****************************************/

template<class T>
class UnstructuredParticleVariable : public UnstructuredParticleVariableBase {
  friend class constUnstructuredVariable<UnstructuredParticleVariableBase, UnstructuredParticleVariable<T>, T, particleIndex>;

public:
  UnstructuredParticleVariable();
  virtual ~UnstructuredParticleVariable();
  UnstructuredParticleVariable(UnstructuredParticleSubset* pset);
  UnstructuredParticleVariable(UnstructuredParticleData<T>*, UnstructuredParticleSubset* pset);
      
    //////////
    // Insert Documentation Here:
  static const UnstructuredTypeDescription* getUnstructuredTypeDescription();

  //////////
  // Insert Documentation Here:
  void resync() {
    d_pdata->resize(getParticleSubset()->numParticles());
  }
      
  //////////
  // Insert Documentation Here:
  virtual UnstructuredParticleVariableBase* clone();
  virtual const UnstructuredParticleVariableBase* clone() const;
  virtual UnstructuredParticleVariableBase* cloneSubset(UnstructuredParticleSubset*);
  virtual const UnstructuredParticleVariableBase* cloneSubset(UnstructuredParticleSubset*) const;

  virtual UnstructuredParticleVariableBase* cloneType() const
  { return scinew UnstructuredParticleVariable<T>(); }
  virtual constUnstructuredParticleVariableBase* cloneConstType() const
  { return scinew constUnstructuredVariable<UnstructuredParticleVariableBase, UnstructuredParticleVariable<T>, T, particleIndex>();
  }

  
  void copyData(const UnstructuredParticleVariable<T>& src);
  virtual void copyData(const UnstructuredParticleVariableBase* src)
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

  virtual void copyPointer(UnstructuredParticleVariable<T>&);
  virtual void copyPointer(UnstructuredVariable&);
  virtual void allocate(UnstructuredParticleSubset*);
  virtual void allocate(int totalParticles);
  virtual void allocate(const UnstructuredPatch*, const IntVector& /*boundary*/)
  { SCI_THROW(InternalError("Should not call UnstructuredParticleVariable<T>::allocate(const UnstructuredPatch*), use allocate(UnstructuredParticleSubset*) instead.", __FILE__, __LINE__)); }

  virtual int size() { return d_pdata->size; }

  // specialized for T=Point
  virtual void gather(UnstructuredParticleSubset* dest,
                      const std::vector<UnstructuredParticleSubset*> &subsets,
                      const std::vector<UnstructuredParticleVariableBase*> &srcs,
                      const std::vector<const UnstructuredPatch*>& /*srcPatches*/,
                      particleIndex extra = 0);  
  virtual void gather(UnstructuredParticleSubset* dest,
                      const std::vector<UnstructuredParticleSubset*> &subsets,
                      const std::vector<UnstructuredParticleVariableBase*> &srcs,
                      particleIndex extra = 0);
  
  virtual void unpackMPI(void* buf, int bufsize, int* bufpos,
                         const ProcessorGroup* pg, UnstructuredParticleSubset* pset);
  virtual void packMPI(void* buf, int bufsize, int* bufpos,
                       const ProcessorGroup* pg, UnstructuredParticleSubset* pset);
  // specialized for T=Point
  virtual void packMPI(void* buf, int bufsize, int* bufpos,
                       const ProcessorGroup* pg, UnstructuredParticleSubset* pset,
                       const UnstructuredPatch* /*forPatch*/);
  virtual void packsizeMPI(int* bufpos,
                           const ProcessorGroup* pg,
                           UnstructuredParticleSubset* pset);
  virtual void emitNormal(std::ostream& out, const IntVector& l,
                          const IntVector& h, ProblemSpecP varnode, bool outputDoubleAsFloat);
  virtual bool emitRLE(std::ostream& out, const IntVector& l, const IntVector& h,
                       ProblemSpecP varnode);
  
  virtual void readNormal(std::istream& in, bool swapBytes);
  virtual void readRLE(std::istream& in, bool swapBytes, int nByteMode);
  
  virtual void* getBasePointer() const;
  virtual const UnstructuredTypeDescription* virtualGetUnstructuredTypeDescription() const;
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

  virtual size_t getDataSize() const {
    return getParticleSubset()->numParticles() * sizeof(T);
  }

  virtual bool copyOut(void* dst) const {
    void* src = (void*)this->getBasePointer();
    size_t numBytes = getDataSize();
    void* retVal = std::memcpy(dst, src, numBytes);
    return (retVal == dst) ? true : false;
  }

protected:
  static UnstructuredTypeDescription* td;
  UnstructuredParticleVariable(const UnstructuredParticleVariable<T>&);
  UnstructuredParticleVariable<T>& operator=(const UnstructuredParticleVariable<T>&);

private:
    //////////
    // Insert Documentation Here:
  UnstructuredParticleData<T>* d_pdata;
  Vector offset_; // only used when T is Point

  static const UnstructuredParticleVariable<T>& castFromBase(const UnstructuredParticleVariableBase* srcptr);
  static UnstructuredVariable* maker();

  // Static variable whose entire purpose is to cause the (instantiated) type of this
  // class to be registered with the Core/Disclosure/TypeDescription class when this
  // class' object code is originally loaded from the shared library.  The 'registerMe'
  // variable is not used for anything else in the program.
  static UnstructuredTypeDescription::Register registerMe;

};

  template<class T>
  UnstructuredTypeDescription* UnstructuredParticleVariable<T>::td = 0;

  // The following line is the initialization (creation) of the 'registerMe' static variable
  // (for each version of UnstructuredParticleVariable (double, int, etc)).  Note, the 'registerMe' variable
  // is created when the object code is initially loaded (usually during intial program load
  // by the operating system).
  template<class T>
  UnstructuredTypeDescription::Register
  UnstructuredParticleVariable<T>::registerMe( getUnstructuredTypeDescription() );

  template<class T>
  const UnstructuredTypeDescription*
  UnstructuredParticleVariable<T>::getUnstructuredTypeDescription()
  {
    if(!td){
      td = scinew UnstructuredTypeDescription(UnstructuredTypeDescription::UnstructuredParticleVariable,
                                  "UnstructuredParticleVariable", &maker,
                                  fun_getUnstructuredTypeDescription((T*)0));
    }
    return td;
  }
   
  template<class T>
  UnstructuredVariable*
  UnstructuredParticleVariable<T>::maker()
  {
    return scinew UnstructuredParticleVariable<T>();
  }
   
  template<class T>
  UnstructuredParticleVariable<T>::UnstructuredParticleVariable()
    : UnstructuredParticleVariableBase(0), d_pdata(0)
  {
  }
   
  template<class T>
  UnstructuredParticleVariable<T>::~UnstructuredParticleVariable()
  {
    if(d_pdata && d_pdata->removeReference())
      delete d_pdata;
  }
   
  template<class T>
  UnstructuredParticleVariable<T>::UnstructuredParticleVariable(UnstructuredParticleSubset* pset)
    : UnstructuredParticleVariableBase(pset)
  {
    d_pdata=scinew UnstructuredParticleData<T>(pset->numParticles());
    d_pdata->addReference();
  }
   
  template<class T>
  void UnstructuredParticleVariable<T>::allocate(int totalParticles)
  {
    ASSERT(isForeign());
    ASSERT(d_pset == 0);

    // this is a pset-less storage as it could have several.  Should be used for
    // foreign data only.  To iterate over particles in this pset, use gather
    d_pdata=scinew UnstructuredParticleData<T>(totalParticles);
    d_pdata->addReference();
  }

  template<class T>
  void UnstructuredParticleVariable<T>::allocate(UnstructuredParticleSubset* pset)
  {
    if (d_pdata && d_pdata->removeReference()) {
      delete d_pdata;
    }
    if (d_pset && d_pset->removeReference()) {
      delete d_pset;
    }

    d_pset = pset;
    d_pset->addReference();
    d_pdata = scinew UnstructuredParticleData<T>(pset->numParticles());
    d_pdata->addReference();
  }
   
  template<class T>
  UnstructuredParticleVariableBase*
  UnstructuredParticleVariable<T>::clone()
  { return scinew UnstructuredParticleVariable<T>(*this); }

  template<class T>
  const UnstructuredParticleVariableBase*
  UnstructuredParticleVariable<T>::clone() const
  { return scinew UnstructuredParticleVariable<T>(*this); }
   
  template<class T>
  UnstructuredParticleVariableBase*
  UnstructuredParticleVariable<T>::cloneSubset(UnstructuredParticleSubset* pset)
  { return scinew UnstructuredParticleVariable<T>(d_pdata, pset); }

  template<class T>
  const UnstructuredParticleVariableBase*
  UnstructuredParticleVariable<T>::cloneSubset(UnstructuredParticleSubset* pset) const
  { return scinew UnstructuredParticleVariable<T>(d_pdata, pset); }

  template<class T>
  const UnstructuredParticleVariable<T>& UnstructuredParticleVariable<T>::castFromBase(const UnstructuredParticleVariableBase* srcptr)
  {
    const UnstructuredParticleVariable<T>* c = dynamic_cast<const UnstructuredParticleVariable<T>* >(srcptr);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in UnstructuredParticle variable", __FILE__, __LINE__));
    return *c;
  }

  template<class T>
    void UnstructuredParticleVariable<T>::copyData(const UnstructuredParticleVariable<T>& src)
  {
    ASSERT(*d_pset == *src.d_pset);
    *d_pdata = *src.d_pdata;
  }


  template<class T>
  UnstructuredParticleVariable<T>::UnstructuredParticleVariable(UnstructuredParticleData<T>* pdata,
                                        UnstructuredParticleSubset* pset)
    : UnstructuredParticleVariableBase(pset), d_pdata(pdata)
  {
    if(d_pdata)
      d_pdata->addReference();
  }
   
  template<class T>
  UnstructuredParticleVariable<T>::UnstructuredParticleVariable(const UnstructuredParticleVariable<T>& copy)
    : UnstructuredParticleVariableBase(copy), d_pdata(copy.d_pdata)
  {
    if(d_pdata)
      d_pdata->addReference();
  }
   
  template<class T>
  void
  UnstructuredParticleVariable<T>::copyPointer(UnstructuredParticleVariable<T>& copy)
  {
    if(this != &copy){
      UnstructuredParticleVariableBase::operator=(copy);
      if(d_pdata && d_pdata->removeReference())
        delete d_pdata;
      d_pdata = copy.d_pdata;
      if(d_pdata)
        d_pdata->addReference();
    }
  }
   
  template<class T>
  void
  UnstructuredParticleVariable<T>::copyPointer(UnstructuredVariable& copy)
  {
    UnstructuredParticleVariable<T>* c = dynamic_cast<UnstructuredParticleVariable<T>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in particle variable", __FILE__, __LINE__));
    copyPointer(*c);
  }
  
  // specialization for T=Point
  template <>
   void UnstructuredParticleVariable<Point>::gather(UnstructuredParticleSubset* pset,
                                       const std::vector<UnstructuredParticleSubset*> &subsets,
                                       const std::vector<UnstructuredParticleVariableBase*> &srcs,
                                       const std::vector<const UnstructuredPatch*>& srcPatches,
                                       particleIndex extra);

  template<class T>
    void UnstructuredParticleVariable<T>::gather(UnstructuredParticleSubset* pset,
                                     const std::vector<UnstructuredParticleSubset*> &subsets,
                                     const std::vector<UnstructuredParticleVariableBase*> &srcs,
                                     const std::vector<const UnstructuredPatch*>& /*srcPatches*/,
                                     particleIndex extra)
  { gather(pset, subsets, srcs, extra); }

template<class T>
  void
  UnstructuredParticleVariable<T>::gather(UnstructuredParticleSubset* pset,
                              const std::vector<UnstructuredParticleSubset*> &subsets,
                              const std::vector<UnstructuredParticleVariableBase*> &srcs,
                              particleIndex extra)
  {
    if(d_pdata && d_pdata->removeReference())
      delete d_pdata;
    if(d_pset && d_pset->removeReference())
      delete d_pset;
    d_pset = pset;
    pset->addReference();
    d_pdata=scinew UnstructuredParticleData<T>(pset->numParticles());
    d_pdata->addReference();
    ASSERTEQ(subsets.size(), srcs.size());
    UnstructuredParticleSubset::iterator dstiter = pset->begin();
    for(int i=0;i<(int)subsets.size();i++){
      UnstructuredParticleVariable<T>* srcptr = dynamic_cast<UnstructuredParticleVariable<T>*>(srcs[i]);
      if(!srcptr)
        SCI_THROW(TypeMismatchException("Type mismatch in UnstructuredParticleVariable::gather", __FILE__, __LINE__));
      UnstructuredParticleVariable<T>& src = *srcptr;
      UnstructuredParticleSubset* subset = subsets[i];
      for(UnstructuredParticleSubset::iterator srciter = subset->begin();
          srciter != subset->end(); srciter++){
        (*this)[*dstiter] = src[*srciter];
        dstiter++;
      }
    }
    ASSERT(dstiter+extra == pset->end());
  }
  
  template<class T>
  void*
  UnstructuredParticleVariable<T>::getBasePointer() const
  {
    return &d_pdata->data[0];
  }
  
  template<class T>
  const UnstructuredTypeDescription*
  UnstructuredParticleVariable<T>::virtualGetUnstructuredTypeDescription() const
  {
    return getUnstructuredTypeDescription();
  }
  
  template<class T>
  void
  UnstructuredParticleVariable<T>::unpackMPI(void* buf, int bufsize, int* bufpos,
                                 const ProcessorGroup* pg,
                                 UnstructuredParticleSubset* pset)
  {
    // This should be fixed for variable sized types!
    const UnstructuredTypeDescription* td = getUnstructuredTypeDescription()->getSubType();
    if(td->isFlat()){
      for(UnstructuredParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        Uintah::MPI::Unpack(buf, bufsize, bufpos,
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
    UnstructuredParticleVariable<Point>::packMPI(void* buf, int bufsize, int* bufpos,
                                     const ProcessorGroup* pg,
                                     UnstructuredParticleSubset* pset,
                                     const UnstructuredPatch* forPatch);
  template<class T>
  void
    UnstructuredParticleVariable<T>::packMPI(void* buf, int bufsize, int* bufpos,
                                 const ProcessorGroup* pg,
                                 UnstructuredParticleSubset* pset,
                                 const UnstructuredPatch* /*forPatch*/)
    { packMPI(buf, bufsize, bufpos, pg, pset); }


  template<class T>
  void
  UnstructuredParticleVariable<T>::packMPI(void* buf, int bufsize, int* bufpos,
                               const ProcessorGroup* pg,
                               UnstructuredParticleSubset* pset)
  {
    // This should be fixed for variable sized types!
    const UnstructuredTypeDescription* td = getUnstructuredTypeDescription()->getSubType();
    if(td->isFlat()){
      for(UnstructuredParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        Uintah::MPI::Pack(&d_pdata->data[*iter], 1, td->getMPIType(),
                 buf, bufsize, bufpos, pg->getComm());
      }
    } else {
      SCI_THROW(InternalError("packMPI not finished\n", __FILE__, __LINE__));
    }
  }

  template<class T>
  void
  UnstructuredParticleVariable<T>::packsizeMPI(int* bufpos,
                                   const ProcessorGroup* pg,
                                   UnstructuredParticleSubset* pset)
  {
    // This should be fixed for variable sized types!
    const UnstructuredTypeDescription* td = getUnstructuredTypeDescription()->getSubType();
    int n = pset->numParticles();
    if(td->isFlat()){
      int size;
      Uintah::MPI::Pack_size(n, td->getMPIType(), pg->getComm(), &size);
      (*bufpos)+= size;
    } else {
      SCI_THROW(InternalError("packsizeMPI not finished\n", __FILE__, __LINE__));
    }
  }

  // Specialized in ParticleVariable_special.cc
  template<>
   void
  UnstructuredParticleVariable<double>::emitNormal(std::ostream& out, const IntVector&,
                                  const IntVector&, ProblemSpecP varnode, bool outputDoubleAsFloat );

  template<class T>
  void
  UnstructuredParticleVariable<T>::emitNormal(std::ostream& out, const IntVector&,
                                  const IntVector&, ProblemSpecP varnode, bool /*outputDoubleAsFloat*/ )
  {
    const UnstructuredTypeDescription* td = fun_getUnstructuredTypeDescription((T*)nullptr);

    if (varnode->findBlock("numParticles") == nullptr) {
      varnode->appendElement("numParticles", d_pset->numParticles());
    }
    if(!td->isFlat()){
      SCI_THROW(InternalError("Cannot yet write non-flat objects!\n", __FILE__, __LINE__));
    }
    else {
      // This could be optimized...
      UnstructuredParticleSubset::iterator iter = d_pset->begin();
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
  UnstructuredParticleVariable<T>::emitRLE(std::ostream& out, const IntVector& /*l*/,
                               const IntVector& /*h*/, ProblemSpecP varnode)
  {
    const UnstructuredTypeDescription* td = fun_getUnstructuredTypeDescription((T*)0);
    if ( varnode->findBlock( "numParticles" ) == nullptr ) {
      varnode->appendElement( "numParticles", d_pset->numParticles() );
    }
    if( !td->isFlat() ){
      SCI_THROW(InternalError( "Cannot yet write non-flat objects!\n", __FILE__, __LINE__) );
    }
    else {
      // emit in runlength encoded format
      RunLengthEncoder<T> rle;
      UnstructuredParticleSubset::iterator iter = d_pset->begin();
      for ( ; iter != d_pset->end(); iter++) {
        rle.addItem((*this)[*iter]);
      }
      rle.write(out);
    }
    return true;
  }
  
  template<class T>
  void
  UnstructuredParticleVariable<T>::readNormal(std::istream& in, bool swapBytes)
  {
    const UnstructuredTypeDescription* td = fun_getUnstructuredTypeDescription((T*)0);
    if(!td->isFlat()) {
      SCI_THROW(InternalError("Cannot yet read non-flat objects!\n", __FILE__, __LINE__));
    }
    else {
      // This could be optimized...
      UnstructuredParticleSubset::iterator iter = d_pset->begin();
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
            swapbytes((*this)[idx]);
          }
        }
      }
    }
  }

  template<class T>
  void
  UnstructuredParticleVariable<T>::readRLE(std::istream& in, bool swapBytes, int nByteMode)
  {
    const UnstructuredTypeDescription* td = fun_getUnstructuredTypeDescription((T*)0);
    if(!td->isFlat()) {
      SCI_THROW(InternalError("Cannot yet read non-flat objects!\n", __FILE__, __LINE__));
    }
    else {
      RunLengthEncoder<T> rle;
      rle.read(in, swapBytes, nByteMode);
      UnstructuredParticleSubset::iterator iter = d_pset->begin();
      typename RunLengthEncoder<T>::iterator rle_iter = rle.begin();
      for ( ; iter != d_pset->end() && rle_iter != rle.end();
            iter++, rle_iter++) {
        (*this)[*iter] = *rle_iter;
      }

      if ((rle_iter != rle.end()) || (iter != d_pset->end())) {
        SCI_THROW(InternalError("UnstructuredParticleVariable::read RLE data is not consistent with the particle subset size", __FILE__, __LINE__));
      }
    }
  }

  template <class T>
  class constUnstructuredParticleVariable : public constUnstructuredVariable<UnstructuredParticleVariableBase, UnstructuredParticleVariable<T>, T, particleIndex>
  {
  public:
    constUnstructuredParticleVariable()
      : constUnstructuredVariable<UnstructuredParticleVariableBase, UnstructuredParticleVariable<T>, T, particleIndex>() {}
    
    constUnstructuredParticleVariable(const UnstructuredParticleVariable<T>& copy)
      : constUnstructuredVariable<UnstructuredParticleVariableBase, UnstructuredParticleVariable<T>, T, particleIndex>(copy) {}

    UnstructuredParticleSubset* getParticleSubset() const {
      return this->rep_.getParticleSubset();
    }
  };

} // End namespace Uintah

#ifdef __PGI
#include <Core/Grid/Variables/UnstructuredParticleVariable_special.cc>
#endif

#endif
