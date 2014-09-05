#ifndef UINTAH_HOMEBREW_CCVARIABLE_H
#define UINTAH_HOMEBREW_CCVARIABLE_H

#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/CCVariableBase.h>
#include <Packages/Uintah/Core/Grid/constGridVariable.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
#include <Packages/Uintah/Core/Grid/SpecializedRunLengthEncoder.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>

#include <unistd.h>

namespace Uintah {

  using namespace SCIRun;

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
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Variable__Cell_Centered

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class T> 
  class CCVariable : public Array3<T>, public CCVariableBase {
    friend class constVariable<CCVariableBase, CCVariable<T>, T, const IntVector&>;
  public:
    CCVariable();
    virtual ~CCVariable();
      
    //////////
    // Insert Documentation Here:
    static const TypeDescription* getTypeDescription();
    
    inline void copyPointer(CCVariable<T>& copy)
    { Array3<T>::copyPointer(copy); }
    
    virtual void copyPointer(CCVariableBase&);

    virtual bool rewindow(const IntVector& low, const IntVector& high)
    { return Array3<T>::rewindow(low, high); }

    // offset the indexing into the array (useful when getting virtual
    // patch data -- i.e. for periodic boundary conditions)
    virtual void offsetGrid(const IntVector& offset)
    { Array3<T>::offset(offset); }
    
    //////////
    // Insert Documentation Here:
    virtual CCVariableBase* clone();
    virtual const CCVariableBase* clone() const;
    virtual CCVariableBase* cloneType() const
    { return scinew CCVariable<T>(); }
    virtual constCCVariableBase* cloneConstType() const
    { return scinew constGridVariable<CCVariableBase, CCVariable<T>, T>(); }

    // Clones the type with a variable having the given extents
    // but with null data -- good as a place holder.
    virtual CCVariableBase* makePlaceHolder(const IntVector & low,
					    const IntVector & high) const
    {
      Array3Window<T>* window = scinew
      Array3Window<T>(0, IntVector(INT_MAX, INT_MAX, INT_MAX), low, high);
      return scinew CCVariable<T>(window);
    }
    
    //////////
    // Insert Documentation Here:
    virtual void allocate(const IntVector& lowIndex,
			  const IntVector& highIndex);
      
    virtual void allocate(const Patch* patch)
    { allocate(patch->getCellLowIndex(), patch->getCellHighIndex()); }
    virtual void allocate(const CCVariable<T>& src)
    { allocate(src.getLowIndex(), src.getHighIndex()); }
    virtual void allocate(const CCVariableBase* src)
    { allocate(castFromBase(src)); }

    //////////
    // Insert Documentation Here:
    void copyPatch(const CCVariable<T>& src,
		   const IntVector& lowIndex, const IntVector& highIndex);
    virtual void copyPatch(const CCVariableBase* src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex)
    { copyPatch(castFromBase(src), lowIndex, highIndex); }
    
    void copyData(const CCVariable<T>& src)
    { copyPatch(src, src.getLowIndex(), src.getHighIndex()); }
    virtual void copyData(const CCVariableBase* src)
    { copyData(castFromBase(src)); }
    
    virtual void* getBasePointer() const;
    virtual const TypeDescription* virtualGetTypeDescription() const;
    virtual void getSizes(IntVector& low, IntVector& high,
			  IntVector& siz) const;
    virtual void getSizes(IntVector& low, IntVector& high,
			  IntVector& dataLow, IntVector& siz,
			  IntVector& strides) const;
    virtual void getSizeInfo(string& elems, unsigned long& totsize,
			     void*& ptr) const {
      IntVector siz = size();
      ostringstream str;
      str << siz.x() << "x" << siz.y() << "x" << siz.z();
      elems=str.str();
      totsize=siz.x()*siz.y()*siz.z()*sizeof(T);
      ptr = (void*)getPointer();
    }
    virtual IntVector getLow()
    { return getLowIndex(); }
    virtual IntVector getHigh()
    { return getHighIndex(); }

    virtual void emitNormal(ostream& out, const IntVector& l, const IntVector& h,
			    ProblemSpecP /*varnode*/)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat())
	Array3<T>::write(out, l, h);
      else
	SCI_THROW(InternalError("Cannot yet write non-flat objects!\n"));
    }

    virtual bool emitRLE(ostream& out, const IntVector& l, const IntVector& h,
			 ProblemSpecP /*varnode*/)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat()){
	RunLengthEncoder<T> rle(Array3<T>::iterator(this, l),
				Array3<T>::iterator(this, h));
	rle.write(out);
      }
      else
	SCI_THROW(InternalError("Cannot yet write non-flat objects!\n"));
      return true;
    }

    virtual void readNormal(istream& in, bool swapBytes)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat())
	Array3<T>::read(in, swapBytes);
      else
	SCI_THROW(InternalError("Cannot yet read non-flat objects!\n"));
    }
      
    virtual void readRLE(istream& in, bool swapBytes, int nByteMode)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat()){
	RunLengthEncoder<T> rle(in, swapBytes, nByteMode);
	rle.copyOut(Array3<T>::begin(), Array3<T>::end());
      }
      else
	SCI_THROW(InternalError("Cannot yet write non-flat objects!\n"));
    }

    static TypeDescription::Register registerMe;

    virtual RefCounted* getRefCounted() {
      return getWindow();
    }
  protected:
    CCVariable(const CCVariable<T>&);

  private:
    CCVariable(Array3Window<T>* window)
      : Array3<T>(window) {}
    CCVariable<T>& operator=(const CCVariable<T>&);

    static const CCVariable<T>& castFromBase(const CCVariableBase* srcptr);
    static Variable* maker();
  };


  template<class T>
  TypeDescription::Register
  CCVariable<T>::registerMe(getTypeDescription());

  template<class T>
  const TypeDescription*
  CCVariable<T>::getTypeDescription()
  {
    static TypeDescription* td;
    if(!td){
      td = scinew TypeDescription(TypeDescription::CCVariable,
				  "CCVariable", &maker,
				  fun_getTypeDescription((T*)0));
    }
    return td;
  }
   
  template<class T>
  const TypeDescription*
  CCVariable<T>::virtualGetTypeDescription() const
  {
    return getTypeDescription();
  }
 
  template<class T>
  Variable*
  CCVariable<T>::maker()
  {
    return scinew CCVariable<T>();
  }
   
  template<class T>
  CCVariable<T>::~CCVariable()
  {
  }
   
  template<class T>
  CCVariableBase*
  CCVariable<T>::clone()
  {
    return scinew CCVariable<T>(*this);
  }

  template<class T>
  const CCVariableBase*
  CCVariable<T>::clone() const
  {
    return scinew CCVariable<T>(*this);
  }

template<class T>
  void
  CCVariable<T>::copyPointer(CCVariableBase& copy)
  {
    CCVariable<T>* c = dynamic_cast<CCVariable<T>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in CC variable"));
    copyPointer(*c);
  }

 
  template<class T>
  CCVariable<T>::CCVariable()
  {
  }

  template<class T>
  CCVariable<T>::CCVariable(const CCVariable<T>& copy)
    : Array3<T>(copy)
  {
  }
   
  template<class T>
  void CCVariable<T>::allocate(const IntVector& lowIndex,
			       const IntVector& highIndex)
  {
    if(getWindow())
      SCI_THROW(InternalError("Allocating a CCvariable that "
			  "is apparently already allocated!"));
    resize(lowIndex, highIndex);
  }
/*
  template<class T>
  void CCVariable<T>::rewindow(const IntVector& low,
			       const IntVector& high) {
    Array3<T> newdata;
    newdata.resize(low, high);
    newdata.copy(*this, low, high);
    resize(low, high);
    Array3<T>::operator=(newdata);
  }
*/

  template<class T>
  const CCVariable<T>& CCVariable<T>::castFromBase(const CCVariableBase* srcptr)
  {
    const CCVariable<T>* c = dynamic_cast<const CCVariable<T>* >(srcptr);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in CC variable"));
    return *c;
  }

  template<class T>
  void
  CCVariable<T>::copyPatch(const CCVariable<T>& src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex)
  {
    if (getWindow()->getData() == src.getWindow()->getData() &&
	getWindow()->getOffset() == src.getWindow()->getOffset()) {
      // No copy needed
      return;
    }

    for(int i=lowIndex.x();i<highIndex.x();i++)
      for(int j=lowIndex.y();j<highIndex.y();j++)
	for(int k=lowIndex.z();k<highIndex.z();k++)
	  (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
  }

  template<class T>
  void*
  CCVariable<T>::getBasePointer() const
  {
    return (void*)getPointer();
  }
  
  template<class T>
  void
  CCVariable<T>::getSizes(IntVector& low, IntVector& high, 
			  IntVector& siz) const
  {
    low = getLowIndex();
    high = getHighIndex();
    siz = size();
  }

  template<class T>
  void
  CCVariable<T>::getSizes(IntVector& low, IntVector& high,
			  IntVector& dataLow, IntVector& siz,
			  IntVector& strides) const
  {
    low=getLowIndex();
    high=getHighIndex();
    dataLow = getWindow()->getOffset();
    siz=size();
    strides = IntVector(sizeof(T), (int)(sizeof(T)*siz.x()),
			(int)(sizeof(T)*siz.y()*siz.x()));
  }

  template <class T>
  class constCCVariable : public constGridVariable<CCVariableBase, CCVariable<T>, T>
  {
  public:
    constCCVariable()
      : constGridVariable<CCVariableBase, CCVariable<T>, T>() {}
    
    constCCVariable(const CCVariable<T>& copy)
      : constGridVariable<CCVariableBase, CCVariable<T>, T>(copy) {}
  };

} // end namespace Uintah

#endif

