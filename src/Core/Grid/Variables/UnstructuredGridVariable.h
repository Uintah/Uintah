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

#ifndef UINTAH_HOMEBREW_UnstructuredGridVARIABLE_H
#define UINTAH_HOMEBREW_UnstructuredGridVARIABLE_H

#include <Core/Grid/Variables/Array1.h>
#include <Core/Grid/Variables/UnstructuredGridVariableBase.h>
#include <Core/Disclosure/UnstructuredTypeDescription.h>
#include <Core/Disclosure/UnstructuredTypeUtils.h>
#include <Core/Grid/UnstructuredPatch.h>
#include <CCA/Ports/InputContext.h>
#include <CCA/Ports/OutputContext.h>
#include <Core/IO/SpecializedRunLengthEncoder.h>
#include <Core/Exceptions/TypeMismatchException.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>
#include <cstring>

namespace Uintah {

  class UnstructuredTypeDescription;

  /**************************************

CLASS
   UnstructuredGridVariable

   Short description...

GENERAL INFORMATION

   UnstructuredGridVariable.h

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
  class UnstructuredGridVariable : public UnstructuredGridVariableBase, public Array1<T> {

  public:
    UnstructuredGridVariable() {}
    virtual ~UnstructuredGridVariable() {}

    inline void copyPointer(UnstructuredGridVariable<T>& copy) { Array1<T>::copyPointer(copy); }

    virtual void copyPointer(UnstructuredVariable&);

    virtual bool rewindow(const IntVector& low, const IntVector& high)
      { return Array1<T>::rewindow(low, high); }

    virtual void offset(const IntVector& offset)  { Array1<T>::offset(offset); }

    // offset the indexing into the array (useful when getting virtual
    // patch data -- i.e. for periodic boundary conditions)
    virtual void offsetGrid(const IntVector& offset) { Array1<T>::offset(offset); }

    static const UnstructuredGridVariable<T>& castFromBase(const UnstructuredGridVariableBase* srcptr);

    //////////
    // Insert Documentation Here:
#if !defined(_AIX)
    using UnstructuredGridVariableBase::allocate; // Quiets PGI compiler warning about hidden virtual function...
#endif
    virtual void allocate(const IntVector& lowIndex, const IntVector& highIndex);

    //////////
    // Insert Documentation Here:
    void copyPatch(const UnstructuredGridVariable<T>& src,
                   const IntVector& lowIndex, const IntVector& highIndex);

    virtual void copyPatch(const UnstructuredGridVariableBase* src,
                           const IntVector& lowIndex,
                           const IntVector& highIndex)
      { copyPatch(castFromBase(src), lowIndex, highIndex); }

    void copyData(const UnstructuredGridVariable<T>& src)
      { copyPatch(src, src.getLowIndex(), src.getHighIndex()); }

    virtual void copyData(const UnstructuredGridVariableBase* src)
      { copyPatch(src, src->getLow(), src->getHigh()); }

    virtual void* getBasePointer() const { return (void*)this->getPointer(); }

    virtual void getSizes(IntVector& low, IntVector& high,
                          IntVector& siz) const;

    virtual void getSizes(IntVector& low, IntVector& high,
                          IntVector& dataLow, IntVector& siz,
                          IntVector& strides) const;

    virtual void getSizeInfo(std::string& elems, unsigned long& totsize,
                             void*& ptr) const {
      IntVector siz = this->size();
      std::ostringstream str;
      str << siz.x() << "x" << siz.y() << "x" << siz.z();
      elems=str.str();
      totsize=siz.x()*siz.y()*siz.z()*sizeof(T);
      ptr = (void*)this->getPointer();
    }

    virtual size_t getDataSize() const {
      IntVector siz = this->size();
      return siz.x() * siz.y() * siz.z() * sizeof(T);
    }

    virtual bool copyOut(void* dst) const {
      void* src = (void*)this->getPointer();
      size_t numBytes = getDataSize();
      void* retVal = std::memcpy(dst, src, numBytes);
      return (retVal == dst) ? true : false;
    }

    virtual IntVector getLow() const {  return this->getLowIndex(); }

    virtual IntVector getHigh() const { return this->getHighIndex(); }

    virtual void emitNormal(std::ostream& out, const IntVector& l, const IntVector& h,
                            ProblemSpecP /*varnode*/, bool outputDoubleAsFloat)
    {
      const UnstructuredTypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat())
        Array1<T>::write(out, l, h, outputDoubleAsFloat);
      else
        SCI_THROW(InternalError("Cannot yet write non-flat objects!\n", __FILE__, __LINE__));
    }

    virtual bool emitRLE(std::ostream& out, const IntVector& l, const IntVector& h,
                         ProblemSpecP /*varnode*/)
    {
      const UnstructuredTypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat()){
        RunLengthEncoder<T> rle;
        Array1<T> & a3 = *this;

        serial_for( a3.range(), [&](int i, int j, int k) {
          rle.addItem( a3(i,j,k) );
        });

        rle.write(out);
      }
      else
        SCI_THROW(InternalError("Cannot yet write non-flat objects!\n", __FILE__, __LINE__));
      return true;
    }

    virtual void readNormal(std::istream& in, bool swapBytes)
    {
      const UnstructuredTypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat())
        Array1<T>::read(in, swapBytes);
      else
        SCI_THROW(InternalError("Cannot yet read non-flat objects!\n", __FILE__, __LINE__));
    }

    virtual void readRLE(std::istream& in, bool swapBytes, int nByteMode)
    {
      const UnstructuredTypeDescription* td = fun_getTypeDescription((T*)0);
      if( td->isFlat() ) {
        RunLengthEncoder<T> rle( in, swapBytes, nByteMode );

        Array1<T> & a3 = *this;
        auto in_itr = rle.begin();
        const auto end_itr = rle.end();

        serial_for( a3.range(), [&](int i, int j, int k) {
          if (in_itr != end_itr) {
            a3(i,j,k) = *in_itr;
            ++in_itr;
          }
        });
      }
      else {
        SCI_THROW(InternalError("Cannot yet write non-flat objects!\n", __FILE__, __LINE__));
      }
    }

    virtual RefCounted* getRefCounted() { return this->getWindow(); }

  protected:
    UnstructuredGridVariable(const UnstructuredGridVariable<T>& copy) : Array1<T>(copy) {}

  private:
    UnstructuredGridVariable(Array1Window<T>* window)
      : Array1<T>(window) {}
    UnstructuredGridVariable<T>& operator=(const UnstructuredGridVariable<T>&);
  };

template<class T>
  void
  UnstructuredGridVariable<T>::copyPointer(UnstructuredVariable& copy)
  {
    UnstructuredGridVariable<T>* c = dynamic_cast<UnstructuredGridVariable<T>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in UnstructuredGrid variable", __FILE__, __LINE__));
    copyPointer(*c);
  }

  template<class T>
  const UnstructuredGridVariable<T>& UnstructuredGridVariable<T>::castFromBase(const UnstructuredGridVariableBase* srcptr)
  {
    const UnstructuredGridVariable<T>* c = dynamic_cast<const UnstructuredGridVariable<T>* >(srcptr);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in CC variable", __FILE__, __LINE__));
    return *c;
  }

  template<class T>
  void UnstructuredGridVariable<T>::allocate( const IntVector& lowIndex,
                                  const IntVector& highIndex )
  {
    if(this->getWindow())
      SCI_THROW(InternalError("Allocating a UnstructuredGridvariable that "
                          "is apparently already allocated!", __FILE__, __LINE__));
    this->resize(lowIndex, highIndex);
  }

  template<class T>
  void
  UnstructuredGridVariable<T>::copyPatch(const UnstructuredGridVariable<T>& src,
                           const IntVector& lowIndex,
                           const IntVector& highIndex)
  {
    if (this->getWindow()->getData() == src.getWindow()->getData() &&
        this->getWindow()->getOffset() == src.getWindow()->getOffset()) {
      // No copy needed
      return;
    }

#if 0
    for(int i=lowIndex.x();i<highIndex.x();i++)
      for(int j=lowIndex.y();j<highIndex.y();j++)
        for(int k=lowIndex.z();k<highIndex.z();k++)
          (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
#endif
    this->copy(src, lowIndex, highIndex);
  }


  template<class T>
  void
  UnstructuredGridVariable<T>::getSizes(IntVector& low, IntVector& high,
                          IntVector& siz) const
  {
    low = this->getLowIndex();
    high = this->getHighIndex();
    siz = this->size();
  }

  template<class T>
  void
  UnstructuredGridVariable<T>::getSizes(IntVector& low, IntVector& high,
                          IntVector& dataLow, IntVector& siz,
                          IntVector& strides) const
  {
    low=this->getLowIndex();
    high=this->getHighIndex();
    dataLow = this->getWindow()->getOffset();
    siz=this->size();
    strides = IntVector(sizeof(T), (int)(sizeof(T)*siz.x()),
                        (int)(sizeof(T)*siz.y()*siz.x()));
  }

} // end namespace Uintah

#endif
