#ifndef UINTAH_HOMEBREW_SFCYVARIABLE_H
#define UINTAH_HOMEBREW_SFCYVARIABLE_H

#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/SFCYVariableBase.h>
#include <Packages/Uintah/Core/Grid/constGridVariable.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
#include <Packages/Uintah/Core/Grid/SpecializedRunLengthEncoder.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Malloc/Allocator.h>
#include <unistd.h>

namespace Uintah {

  using namespace SCIRun;

  class TypeDescription;

  /**************************************

CLASS
   SFCYVariable
   
GENERAL INFORMATION

   SFCYVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SFCYVariable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class T>
  class SFCYVariable : public Array3<T>, public SFCYVariableBase {
    friend class constVariable<SFCYVariableBase, SFCYVariable<T>, T, const IntVector&>;
  public:
     
    SFCYVariable();
    virtual ~SFCYVariable();
     
    //////////
    // Insert Documentation Here:
    static const TypeDescription* getTypeDescription();
     
    inline void copyPointer(SFCYVariable<T>& copy)
    { Array3<T>::copyPointer(copy); }

    virtual void copyPointer(SFCYVariableBase&);

    virtual bool rewindow(const IntVector& low, const IntVector& high)
    { return Array3<T>::rewindow(low, high); }

    // offset the indexing into the array (useful when getting virtual
    // patch data -- i.e. for periodic boundary conditions)
    virtual void offsetGrid(const IntVector& offset)
    { Array3<T>::offset(offset); }
    
    //////////
    // Insert Documentation Here:
    virtual SFCYVariableBase* clone();
    virtual const SFCYVariableBase* clone() const;    
    virtual SFCYVariableBase* cloneType() const
    { return scinew SFCYVariable<T>(); }
    virtual constSFCYVariableBase* cloneConstType() const
    { return scinew constGridVariable<SFCYVariableBase, SFCYVariable<T>, T>();
    }

    // Clones the type with a variable having the given extents
    // but with null data -- good as a place holder.
    virtual SFCYVariableBase* makePlaceHolder( const IntVector & low,
					       const IntVector & high ) const
    {
      Array3Window<T>* window = scinew
      Array3Window<T>(0, IntVector(INT_MAX, INT_MAX, INT_MAX), low, high);
      return scinew SFCYVariable<T>(window);
    }
    
    //////////
    // Insert Documentation Here:
    virtual void allocate(const IntVector& lowIndex,
			  const IntVector& highIndex);
     
    virtual void allocate(const Patch* patch)
    { allocate(patch->getSFCYLowIndex(), patch->getSFCYHighIndex()); }
    virtual void allocate(const SFCYVariable<T>& src)
    { allocate(src.getLowIndex(), src.getHighIndex()); }
    virtual void allocate(const SFCYVariableBase* src)
    { allocate(castFromBase(src)); }

    void copyPatch(const SFCYVariable<T>& src,
		   const IntVector& lowIndex, const IntVector& highIndex);
    virtual void copyPatch(const SFCYVariableBase* src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex)
    { copyPatch(castFromBase(src), lowIndex, highIndex); }
    
    void copyData(const SFCYVariable<T>& src)
    { copyPatch(src, src.getLowIndex(), src.getHighIndex()); }
    virtual void copyData(const SFCYVariableBase* src)
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
    SFCYVariable(const SFCYVariable<T>&);
  private:
    SFCYVariable(Array3Window<T>* window)
      : Array3<T>(window) {}
    SFCYVariable<T>& operator=(const SFCYVariable<T>&);

    static const SFCYVariable<T>& castFromBase(const SFCYVariableBase* srcptr);
    static Variable* maker();
  };
   
  template<class T>
  TypeDescription::Register SFCYVariable<T>::registerMe(getTypeDescription());

  template<class T>
  const TypeDescription*
  SFCYVariable<T>::getTypeDescription()
  {
    static TypeDescription* td;
    if(!td){
      td = scinew TypeDescription(TypeDescription::SFCYVariable,
				  "SFCYVariable", &maker,
				  fun_getTypeDescription((T*)0));
    }
    return td;
  }
   
  template<class T>
  Variable*
  SFCYVariable<T>::maker()
  {
    return scinew SFCYVariable<T>();
  }
   
  template<class T>
  SFCYVariable<T>::~SFCYVariable()
  {
  }
   
  template<class T>
  SFCYVariableBase*
  SFCYVariable<T>::clone()
  {
    return scinew SFCYVariable<T>(*this);
  }

  template<class T>
  const SFCYVariableBase*
  SFCYVariable<T>::clone() const
  {
    return scinew SFCYVariable<T>(*this);
  }
   
  template<class T>
  void
  SFCYVariable<T>::copyPointer(SFCYVariableBase& copy)
  {
    SFCYVariable<T>* c = dynamic_cast<SFCYVariable<T>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in SFCY variable"));
    copyPointer(*c);
  }

  template<class T>
  SFCYVariable<T>::SFCYVariable()
  {
  }

  template<class T>
  SFCYVariable<T>::SFCYVariable(const SFCYVariable<T>& copy)
    : Array3<T>(copy)
  {
  }
   
  template<class T>
  void
  SFCYVariable<T>::allocate(const IntVector& lowIndex,
			    const IntVector& highIndex)
  {
    if(getWindow())
      SCI_THROW(InternalError("Allocating an SFCYvariable that "
			  "is apparently already allocated!"));
    resize(lowIndex, highIndex);
  }
/*
  template<class T>
  void SFCYVariable<T>::rewindow(const IntVector& low,
				 const IntVector& high) {
    Array3<T> newdata;
    newdata.resize(low, high);
    newdata.copy(*this, low, high);
    resize(low, high);
    Array3<T>::operator=(newdata);
  }
*/

  template<class T>
  const SFCYVariable<T>& SFCYVariable<T>::castFromBase(const SFCYVariableBase* srcptr)
  {
    const SFCYVariable<T>* c = dynamic_cast<const SFCYVariable<T>* >(srcptr);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in SFCY variable"));
    return *c;
  }

  template<class T>
  void
  SFCYVariable<T>::copyPatch(const SFCYVariable<T>& src,
			     const IntVector& lowIndex,
			     const IntVector& highIndex)
  {
    if (getWindow()->getData() == src.getWindow()->getData() &&
	getWindow()->getOffset() == src.getWindow()->getOffset()) {
      // No copy needed
      //cerr << "No copy needed for SFCYVariable!!!\n";
      return;
    }
    
    for(int i=lowIndex.x();i<highIndex.x();i++)
      for(int j=lowIndex.y();j<highIndex.y();j++)
	for(int k=lowIndex.z();k<highIndex.z();k++)
	  (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
  }
   
  template<class T>
  void*
  SFCYVariable<T>::getBasePointer() const
  {
    return (void*)getPointer();
  }

  template<class T>
  const TypeDescription*
  SFCYVariable<T>::virtualGetTypeDescription() const
  {
    return getTypeDescription();
  }
   
  template<class T>
  void
  SFCYVariable<T>::getSizes(IntVector& low, IntVector& high, 
			    IntVector& siz) const
  {
    low = getLowIndex();
    high = getHighIndex();
    siz = size();
  }
  template<class T>
  void
  SFCYVariable<T>::getSizes(IntVector& low, IntVector& high,
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
  class constSFCYVariable : public constGridVariable<SFCYVariableBase, SFCYVariable<T>, T>
  {
  public:
    constSFCYVariable()
      : constGridVariable<SFCYVariableBase, SFCYVariable<T>, T>() {}
    
    constSFCYVariable(const SFCYVariable<T>& copy)
      : constGridVariable<SFCYVariableBase, SFCYVariable<T>, T>(copy) {}
  };

} // end namespace Uintah

#endif
