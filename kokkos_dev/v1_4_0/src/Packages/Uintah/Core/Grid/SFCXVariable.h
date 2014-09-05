#ifndef UINTAH_HOMEBREW_SFCXVARIABLE_H
#define UINTAH_HOMEBREW_SFCXVARIABLE_H

#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/SFCXVariableBase.h>
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
   SFCXVariable
   
GENERAL INFORMATION

   SFCXVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SFCXVariable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class T>
  class SFCXVariable : public Array3<T>, public SFCXVariableBase {
    friend class constVariable<SFCXVariableBase, SFCXVariable<T>, T, const IntVector&>;
  public:
     
    SFCXVariable();
    virtual ~SFCXVariable();
     
    //////////
    // Insert Documentation Here:
    static const TypeDescription* getTypeDescription();

    inline void copyPointer(SFCXVariable<T>& copy)
    { Array3<T>::copyPointer(copy); }

    virtual void copyPointer(SFCXVariableBase&);

    virtual bool rewindow(const IntVector& low, const IntVector& high)
    { return Array3<T>::rewindow(low, high); }

    // offset the indexing into the array (useful when getting virtual
    // patch data -- i.e. for periodic boundary conditions)
    virtual void offsetGrid(IntVector offset)
    { Array3<T>::offset(offset); }
    
    //////////
    // Insert Documentation Here:
    virtual SFCXVariableBase* clone();
    virtual const SFCXVariableBase* clone() const;
    virtual SFCXVariableBase* cloneType() const
    { return scinew SFCXVariable<T>(); }
    virtual constSFCXVariableBase* cloneConstType() const
    { return scinew constGridVariable<SFCXVariableBase, SFCXVariable<T>, T>();
    }

    //////////
    // Insert Documentation Here:
    virtual void allocate(const IntVector& lowIndex,
			  const IntVector& highIndex);

    virtual void allocate(const Patch* patch)
    { allocate(patch->getSFCXLowIndex(), patch->getSFCXHighIndex()); }
     virtual void allocate(const SFCXVariable<T>& src)
    { allocate(src.getLowIndex(), src.getHighIndex()); }
    virtual void allocate(const SFCXVariableBase* src)
    { allocate(castFromBase(src)); }
     
    void copyPatch(const SFCXVariable<T>& src,
		   const IntVector& lowIndex, const IntVector& highIndex);
    virtual void copyPatch(const SFCXVariableBase* src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex)
    { copyPatch(castFromBase(src), lowIndex, highIndex); }
    
    void copyData(const SFCXVariable<T>& src)
    { copyPatch(src, src.getLowIndex(), src.getHighIndex()); }
    virtual void copyData(const SFCXVariableBase* src)
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

    // Replace the values on the indicated face with value
   
    
    virtual void emitNormal(ostream& out, DOM_Element /*varnode*/)
      {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat())
	Array3<T>::write(out);
      else
	throw InternalError("Cannot yet write non-flat objects!\n");
    }
  
    virtual bool emitRLE(ostream& out, DOM_Element /*varnode*/)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat()){
	RunLengthEncoder<T> rle(Array3<T>::begin(), Array3<T>::end());
	rle.write(out);
      }
      else
	throw InternalError("Cannot yet write non-flat objects!\n");
      return true;
    }
  
    virtual void readNormal(istream& in)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat())
	Array3<T>::read(in);
      else
	throw InternalError("Cannot yet read non-flat objects!\n");
    }
  
    virtual void readRLE(istream& in)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat()){
	RunLengthEncoder<T> rle(in);
	rle.copyOut(Array3<T>::begin(), Array3<T>::end());
      }
      else
	throw InternalError("Cannot yet write non-flat objects!\n");
    }
 
    static TypeDescription::Register registerMe;
    virtual RefCounted* getRefCounted() {
      return getWindow();
    }

  protected:
    SFCXVariable(const SFCXVariable<T>&);
  private:
    SFCXVariable<T>& operator=(const SFCXVariable<T>&);
    
    static const SFCXVariable<T>& castFromBase(const SFCXVariableBase* srcptr);
    static Variable* maker();
  };
   
  template<class T>
  TypeDescription::Register SFCXVariable<T>::registerMe(getTypeDescription());

  template<class T>
  const TypeDescription*
  SFCXVariable<T>::getTypeDescription()
  {
    static TypeDescription* td;
    if(!td){
      td = scinew TypeDescription(TypeDescription::SFCXVariable,
				  "SFCXVariable", &maker,
				  fun_getTypeDescription((T*)0));
    }
    return td;
  }
   
  template<class T>
  Variable*
  SFCXVariable<T>::maker()
  {
    return scinew SFCXVariable<T>();
  }
   
  template<class T>
  SFCXVariable<T>::~SFCXVariable()
  {
  }
   
  template<class T>
  SFCXVariableBase*
  SFCXVariable<T>::clone()
  {
    return scinew SFCXVariable<T>(*this);
  }

  template<class T>
  const SFCXVariableBase*
  SFCXVariable<T>::clone() const
  {
    return scinew SFCXVariable<T>(*this);
  }

  template<class T>
  void
  SFCXVariable<T>::copyPointer(SFCXVariableBase& copy)
  {
    SFCXVariable<T>* c = dynamic_cast<SFCXVariable<T>* >(&copy);
    if(!c)
      throw TypeMismatchException("Type mismatch in SFCX variable");
    copyPointer(*c);   
  }

  template<class T>
  SFCXVariable<T>::SFCXVariable()
  {
  }

  template<class T>
  SFCXVariable<T>::SFCXVariable(const SFCXVariable<T>& copy)
    : Array3<T>(copy)
  {
  }

  template<class T>
  void
  SFCXVariable<T>::allocate(const IntVector& lowIndex,
			    const IntVector& highIndex)
  {
    if(getWindow())
      throw InternalError("Allocating an SFCXvariable that "
			  "is apparently already allocated!");
    resize(lowIndex, highIndex);
  }

  template<class T>
  const SFCXVariable<T>& SFCXVariable<T>::castFromBase(const SFCXVariableBase* srcptr)
  {
    const SFCXVariable<T>* c = dynamic_cast<const SFCXVariable<T>* >(srcptr);
    if(!c)
      throw TypeMismatchException("Type mismatch in SFCX variable");
    return *c;
  }

  template<class T>
  void
  SFCXVariable<T>::copyPatch(const SFCXVariable<T>& src,
			     const IntVector& lowIndex,
			     const IntVector& highIndex)
  {
    for(int i=lowIndex.x();i<highIndex.x();i++)
      for(int j=lowIndex.y();j<highIndex.y();j++)
	for(int k=lowIndex.z();k<highIndex.z();k++)
	  (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
  }
   
  template<class T>
  void*
  SFCXVariable<T>::getBasePointer() const
  {
    return (void*)getPointer();
  }

  template<class T>
  const TypeDescription*
  SFCXVariable<T>::virtualGetTypeDescription() const
  {
    return getTypeDescription();
  }
  template<class T>
  void
  SFCXVariable<T>::getSizes(IntVector& low, IntVector& high, 
			    IntVector& siz) const
  {
    low = getLowIndex();
    high = getHighIndex();
    siz = size();
  }
  template<class T>
  void
  SFCXVariable<T>::getSizes(IntVector& low, IntVector& high,
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
  class constSFCXVariable : public constGridVariable<SFCXVariableBase, SFCXVariable<T>, T>
  {
  public:
    constSFCXVariable()
      : constGridVariable<SFCXVariableBase, SFCXVariable<T>, T>() {}
    
    constSFCXVariable(const SFCXVariable<T>& copy)
      : constGridVariable<SFCXVariableBase, SFCXVariable<T>, T>(copy) {}
  };

} // end namespace Uintah

#endif
