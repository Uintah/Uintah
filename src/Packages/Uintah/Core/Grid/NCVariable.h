#ifndef UINTAH_HOMEBREW_NCVARIABLE_H
#define UINTAH_HOMEBREW_NCVARIABLE_H

#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/NCVariableBase.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/constGridVariable.h>
#include <Packages/Uintah/Core/Grid/SpecializedRunLengthEncoder.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>

#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>

#include <Core/Exceptions/InternalError.h>

#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>

#include <unistd.h>

namespace Uintah {

  class TypeDescription;

  using SCIRun::IntVector;
  using SCIRun::InternalError;
  using SCIRun::RunLengthEncoder;

  /**************************************

CLASS
   NCVariable
   
GENERAL INFORMATION

   NCVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   NCVariable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class T>
  class NCVariable : public Array3<T>, public NCVariableBase {
    friend class constVariable<NCVariableBase, NCVariable<T>, T, const IntVector&>;
  public:
     
    NCVariable();
    virtual ~NCVariable();
     
    //////////
    // Insert Documentation Here:
    static const TypeDescription* getTypeDescription();
    
    inline void copyPointer(NCVariable<T>& copy)
    { Array3<T>::copyPointer(copy); }

    virtual void copyPointer(NCVariableBase&);

    virtual bool rewindow(const IntVector& low, const IntVector& high)
    { return Array3<T>::rewindow(low, high); }    

    // offset the indexing into the array (useful when getting virtual
    // patch data -- i.e. for periodic boundary conditions)
    virtual void offsetGrid(const IntVector& offset)
    { Array3<T>::offset(offset); }    
    
    //////////
    // Insert Documentation Here:
    virtual NCVariableBase* clone();
    virtual const NCVariableBase* clone() const;    
    virtual NCVariableBase* cloneType() const
    { return scinew NCVariable<T>(); }
    virtual constNCVariableBase* cloneConstType() const
    { return scinew constGridVariable<NCVariableBase, NCVariable<T>, T>(); }

    // Clones the type with a variable having the given extents
    // but with null data -- good as a place holder.
    virtual NCVariableBase* makePlaceHolder(const IntVector & low,
					    const IntVector & high) const
    {
      Array3Window<T>* window = scinew
      Array3Window<T>(0, IntVector(INT_MAX, INT_MAX, INT_MAX), low, high);
      return scinew NCVariable<T>(window);
    }
    
    //////////
    // Insert Documentation Here:
    virtual void allocate(const IntVector& lowIndex,
			  const IntVector& highIndex);
     
    virtual void allocate(const Patch* patch)
    { allocate(patch->getNodeLowIndex(), patch->getNodeHighIndex()); }
    virtual void allocate(const NCVariable<T>& src)
    { allocate(src.getLowIndex(), src.getHighIndex()); }
    virtual void allocate(const NCVariableBase* src)
    { allocate(castFromBase(src)); }

    void copyPatch(const NCVariable<T>& src,
		   const IntVector& lowIndex,
		   const IntVector& highIndex);
    virtual void copyPatch(const NCVariableBase* src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex)
    { copyPatch(castFromBase(src), lowIndex, highIndex); }
    
    void copyData(const NCVariable<T>& src)
    { copyPatch(src, src.getLowIndex(), src.getHighIndex()); }
    virtual void copyData(const NCVariableBase* src)
    { copyData(castFromBase(src)); }
    
    virtual void* getBasePointer() const;
    virtual const TypeDescription* virtualGetTypeDescription() const;

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
    NCVariable(const NCVariable<T>&);
  private:
    NCVariable(Array3Window<T>* window)
      : Array3<T>(window) {}
    NCVariable<T>& operator=(const NCVariable<T>&);

    static const NCVariable<T>& castFromBase(const NCVariableBase* srcptr);
    static Variable* maker();
  };
   
  template<class T>
  TypeDescription::Register NCVariable<T>::registerMe(getTypeDescription());

  template<class T>
  const TypeDescription*
  NCVariable<T>::getTypeDescription()
  {
    static TypeDescription* td;
    if(!td){
      td = scinew TypeDescription(TypeDescription::NCVariable,
				  "NCVariable", &maker,
				  fun_getTypeDescription((T*)0));
    }
    return td;
  }
   
  template<class T>
  Variable*
  NCVariable<T>::maker()
  {
    return scinew NCVariable<T>();
  }
   
  template<class T>
  NCVariable<T>::~NCVariable()
  {
  }
   
  template<class T>
  NCVariableBase*
  NCVariable<T>::clone()
  {
    NCVariable<T>* tmp=scinew NCVariable<T>(*this);
    return tmp;
  }

  template<class T>
  const NCVariableBase*
  NCVariable<T>::clone() const
  {
    NCVariable<T>* tmp=scinew NCVariable<T>(*this);
    return tmp;
  }
   
  template<class T>
  void
  NCVariable<T>::copyPointer(NCVariableBase& copy)
  {
    NCVariable<T>* c = dynamic_cast<NCVariable<T>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in NC variable"));
    copyPointer(*c);
  }

  template<class T>
  NCVariable<T>::NCVariable()
  {
  }

  template<class T>
  NCVariable<T>::NCVariable(const NCVariable<T>& copy)
    : Array3<T>(copy)
  {
  }

  template<class T>
  void
  NCVariable<T>::allocate(const IntVector& lowIndex,
			  const IntVector& highIndex)
  {
    if(getWindow())
      SCI_THROW(InternalError("Allocating an NCvariable that "
			  "is apparently already allocated!"));
    resize(lowIndex, highIndex);
  }

  template<class T>
  const NCVariable<T>& NCVariable<T>::castFromBase(const NCVariableBase* srcptr)
  {
    const NCVariable<T>* c = dynamic_cast<const NCVariable<T>* >(srcptr);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in NC variable"));
    return *c;
  }

  template<class T>
  void
  NCVariable<T>::copyPatch(const NCVariable<T>& src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex)
  {
     if (getWindow()->getData() == src.getWindow()->getData() &&
	getWindow()->getOffset() == src.getWindow()->getOffset()) {
      // No copy needed
       //cerr << "No copy needed for NCVariable!!!\n";
      return;
    }

    for(int i=lowIndex.x();i<highIndex.x();i++)
      for(int j=lowIndex.y();j<highIndex.y();j++)
	for(int k=lowIndex.z();k<highIndex.z();k++)
	  (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
  }
   
  template<class T>
  void*
  NCVariable<T>::getBasePointer() const
  {
    return (void*)getPointer();
  }

  template<class T>
  const TypeDescription*
  NCVariable<T>::virtualGetTypeDescription() const
  {
    return getTypeDescription();
  }
   
  template<class T>
  void
  NCVariable<T>::getSizes(IntVector& low, IntVector& high,
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
  class constNCVariable : public constGridVariable<NCVariableBase, NCVariable<T>, T>
  {
  public:
    constNCVariable()
      : constGridVariable<NCVariableBase, NCVariable<T>, T>() {}
    
    constNCVariable(const NCVariable<T>& copy)
      : constGridVariable<NCVariableBase, NCVariable<T>, T>(copy) {}
  };

} // end namespace Uintah
#endif
