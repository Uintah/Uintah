#ifndef UINTAH_HOMEBREW_SFCZVARIABLE_H
#define UINTAH_HOMEBREW_SFCZVARIABLE_H

#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariableBase.h>
#include <Packages/Uintah/Core/Grid/Variables/constGridVariable.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
#include <Packages/Uintah/Core/IO/SpecializedRunLengthEncoder.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Malloc/Allocator.h>
#include <unistd.h>

namespace Uintah {

  using SCIRun::InternalError;

  class TypeDescription;

  /**************************************

CLASS
   SFCZVariable
   
GENERAL INFORMATION

   SFCZVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SFCZVariable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class T>
  class SFCZVariable : public Array3<T>, public SFCZVariableBase {
    friend class constVariable<SFCZVariableBase, SFCZVariable<T>, T, const IntVector&>;
  public:
     
    SFCZVariable();
    virtual ~SFCZVariable();
     
    //////////
    // Insert Documentation Here:
    static const TypeDescription* getTypeDescription();
     
    inline void copyPointer(SFCZVariable<T>& copy)
    { Array3<T>::copyPointer(copy); }

    virtual void copyPointer(Variable&);

    virtual bool rewindow(const IntVector& low, const IntVector& high)
    { return Array3<T>::rewindow(low, high); }

    virtual void offset(const IntVector& offset)
    { Array3<T>::offset(offset); } 

    // offset the indexing into the array (useful when getting virtual
    // patch data -- i.e. for periodic boundary conditions)
    virtual void offsetGrid(const IntVector& offset)
    { Array3<T>::offset(offset); }

    //////////
    // Insert Documentation Here:
    virtual GridVariable* clone();
    virtual const GridVariable* clone() const;    
    virtual SFCZVariableBase* cloneType() const
    { return scinew SFCZVariable<T>(); }
    virtual constSFCZVariableBase* cloneConstType() const
    { return scinew constGridVariable<SFCZVariableBase, SFCZVariable<T>, T>();
    }

    // Clones the type with a variable having the given extents
    // but with null data -- good as a place holder.
    virtual GridVariable* makePlaceHolder(const IntVector & low,
					      const IntVector & high) const
    {
      Array3Window<T>* window = scinew
      Array3Window<T>(0, IntVector(INT_MAX, INT_MAX, INT_MAX), low, high);
      return scinew SFCZVariable<T>(window);
    }    

    //////////
    // Insert Documentation Here:
    virtual void allocate(const IntVector& lowIndex,
			  const IntVector& highIndex);
     
    virtual void allocate(const Patch* patch, const IntVector& boundary)
    {      
      IntVector l,h;
      patch->computeVariableExtents(Patch::ZFaceBased, boundary, 
                                    Ghost::None, 0, l, h);
      allocate(l, h);
    }
    virtual void allocate(const SFCZVariable<T>& src)
    { allocate(src.getLowIndex(), src.getHighIndex()); }
    virtual void allocate(const GridVariable* src)
    { allocate(castFromBase(src)); }

    void copyPatch(const SFCZVariable<T>& src,
		   const IntVector& lowIndex, const IntVector& highIndex);
    virtual void copyPatch(const GridVariable* src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex)
    { copyPatch(castFromBase(src), lowIndex, highIndex); }
    
    void copyData(const SFCZVariable<T>& src)
    { copyPatch(src, src.getLowIndex(), src.getHighIndex()); }
    virtual void copyData(const GridVariable* src)
    { copyData(castFromBase(src)); }
     
    virtual void* getBasePointer() const;
    virtual const TypeDescription* virtualGetTypeDescription() const;

    // If the window is the same size as its data then dataLow == low,
    // otherwise low may be offset from dataLow.
    virtual void getSizes(IntVector& low, IntVector& high,
			  IntVector& siz) const;
    virtual void getSizes(IntVector& low, IntVector& high,
			  IntVector& dataLow, IntVector& siz,
			  IntVector& strides) const;
    virtual void getSizeInfo(string& elems, unsigned long& totsize,
			     void*& ptr) const {
      IntVector siz = this->size();
      ostringstream str;
      str << siz.x() << "x" << siz.y() << "x" << siz.z();
      elems=str.str();
      totsize=siz.x()*siz.y()*siz.z()*sizeof(T);
      ptr = (void*)this->getPointer();
    }
    virtual IntVector getLow()
    { return this->getLowIndex(); }
    virtual IntVector getHigh()
    { return this->getHighIndex(); }

    // Replace the values on the indicated face with value
   
    virtual void emitNormal(ostream& out, const IntVector& l,
			    const IntVector& h, ProblemSpecP /*varnode*/, bool outputDoubleAsFloat)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat())
	Array3<T>::write(out, l, h, outputDoubleAsFloat);
      else
	SCI_THROW(InternalError("Cannot yet write non-flat objects!\n"));
    }
      
    virtual bool emitRLE(ostream& out, const IntVector& l,
			 const IntVector& h, ProblemSpecP /*varnode*/)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat()){
	SCIRun::RunLengthEncoder<T> rle(typename Array3<T>::iterator(this, l),
                                        typename Array3<T>::iterator(this, h));
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
	SCIRun::RunLengthEncoder<T> rle(in, swapBytes, nByteMode);
	rle.copyOut(Array3<T>::begin(), Array3<T>::end());
      }
      else
	SCI_THROW(InternalError("Cannot yet write non-flat objects!\n"));
    }

    static TypeDescription::Register registerMe;
    virtual RefCounted* getRefCounted() {
      return this->getWindow();
    }
  protected:
    SFCZVariable(const SFCZVariable<T>&);
  private:
    SFCZVariable(Array3Window<T>* window)
      : Array3<T>(window) {}
    SFCZVariable<T>& operator=(const SFCZVariable<T>&);

    static const SFCZVariable<T>& castFromBase(const GridVariable* srcptr);
    static Variable* maker();
  };
   
  template<class T>
  TypeDescription::Register SFCZVariable<T>::registerMe(getTypeDescription());

  template<class T>
  const TypeDescription*
  SFCZVariable<T>::getTypeDescription()
  {
    static TypeDescription* td;
    if(!td){
      td = scinew TypeDescription(TypeDescription::SFCZVariable,
				  "SFCZVariable", &maker,
				  fun_getTypeDescription((T*)0));
    }
    return td;
  }
   
  template<class T>
  Variable*
  SFCZVariable<T>::maker()
  {
    return scinew SFCZVariable<T>();
  }
   
  template<class T>
  SFCZVariable<T>::~SFCZVariable()
  {
  }
   
  template<class T>
  GridVariable*
  SFCZVariable<T>::clone()
  {
    return scinew SFCZVariable<T>(*this);
  }

  template<class T>
  const GridVariable*
  SFCZVariable<T>::clone() const
  {
    return scinew SFCZVariable<T>(*this);
  }
   
  template<class T>
  void
  SFCZVariable<T>::copyPointer(Variable& copy)
  {
    SFCZVariable<T>* c = dynamic_cast<SFCZVariable<T>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in SFCZ variable"));
    copyPointer(*c);
  }

  template<class T>
  SFCZVariable<T>::SFCZVariable()
  {
  }

  template<class T>
  SFCZVariable<T>::SFCZVariable(const SFCZVariable<T>& copy)
    : Array3<T>(copy)
  {
  }
   
  template<class T>
  void
  SFCZVariable<T>::allocate(const IntVector& lowIndex,
			    const IntVector& highIndex)
  {
    if(this->getWindow())
      SCI_THROW(InternalError("Allocating an SFCZvariable that "
			  "is apparently already allocated!"));
    this->resize(lowIndex, highIndex);
  }
/*
  template<class T>
  void SFCZVariable<T>::rewindow(const IntVector& low,
				 const IntVector& high) {
    Array3<T> newdata;
    newdata.resize(low, high);
    newdata.copy(*this, low, high);
    this->resize(low, high);
    Array3<T>::operator=(newdata);
  }
*/

  template<class T>
  const SFCZVariable<T>& SFCZVariable<T>::castFromBase(const GridVariable* srcptr)
  {
    const SFCZVariable<T>* c = dynamic_cast<const SFCZVariable<T>* >(srcptr);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in SFCZ variable"));
    return *c;
  }

  template<class T>
  void
  SFCZVariable<T>::copyPatch(const SFCZVariable<T>& src,
			     const IntVector& lowIndex,
			     const IntVector& highIndex)
  {
    if (this->getWindow()->getData() == src.getWindow()->getData() &&
	this->getWindow()->getOffset() == src.getWindow()->getOffset()) {
      // No copy needed
      //cerr << "No copy needed for SFCZVariable!!!\n";
      return;
    }
    
    for(int i=lowIndex.x();i<highIndex.x();i++)
      for(int j=lowIndex.y();j<highIndex.y();j++)
	for(int k=lowIndex.z();k<highIndex.z();k++)
	  (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
  }
   
  template<class T>
  void*
  SFCZVariable<T>::getBasePointer() const
  {
    return (void*)this->getPointer();
  }

  template<class T>
  const TypeDescription*
  SFCZVariable<T>::virtualGetTypeDescription() const
  {
    return getTypeDescription();
  }
  template<class T>
  void
  SFCZVariable<T>::getSizes(IntVector& low, IntVector& high, 
			    IntVector& siz) const
  {
    low = this->getLowIndex();
    high = this->getHighIndex();
    siz = this->size();
  }

  template<class T>
  void
  SFCZVariable<T>::getSizes(IntVector& low, IntVector& high,
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

  template <class T>
  class constSFCZVariable : public constGridVariable<SFCZVariableBase, SFCZVariable<T>, T>
  {
  public:
    constSFCZVariable()
      : constGridVariable<SFCZVariableBase, SFCZVariable<T>, T>() {}
    
    constSFCZVariable(const SFCZVariable<T>& copy)
      : constGridVariable<SFCZVariableBase, SFCZVariable<T>, T>(copy) {}
  };

} // end namespace Uintah

#endif
