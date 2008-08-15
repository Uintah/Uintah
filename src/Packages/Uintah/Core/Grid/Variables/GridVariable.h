#ifndef UINTAH_HOMEBREW_GridVARIABLE_H
#define UINTAH_HOMEBREW_GridVARIABLE_H

#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <Packages/Uintah/Core/Grid/Variables/GridVariableBase.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
#include <Packages/Uintah/Core/IO/SpecializedRunLengthEncoder.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace Uintah {

  using SCIRun::InternalError;

  class TypeDescription;

  /**************************************

CLASS
   GridVariable
   
   Short description...

GENERAL INFORMATION

   GridVariable.h

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
  class GridVariable : public GridVariableBase, public Array3<T> {
  public:
    GridVariable() {}
    virtual ~GridVariable() {}
      
    inline void copyPointer(GridVariable<T>& copy)
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
    
    static const GridVariable<T>& castFromBase(const GridVariableBase* srcptr);

    //////////
    // Insert Documentation Here:
    using GridVariableBase::allocate; // Quiets PGI compiler warning about hidden virtual function...
    virtual void allocate(const IntVector& lowIndex, const IntVector& highIndex);
      
    //////////
    // Insert Documentation Here:
    void copyPatch(const GridVariable<T>& src,
		   const IntVector& lowIndex, const IntVector& highIndex);
    virtual void copyPatch(const GridVariableBase* src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex)
    { copyPatch(castFromBase(src), lowIndex, highIndex); }
    
    void copyData(const GridVariable<T>& src)
    { copyPatch(src, src.getLowIndex(), src.getHighIndex()); }
    virtual void copyData(const GridVariableBase* src)
    { copyPatch(src, src->getLow(), src->getHigh()); }
    
    virtual void* getBasePointer() const 
    { return (void*)this->getPointer(); }

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
    virtual IntVector getLow() const
    { return this->getLowIndex(); }
    virtual IntVector getHigh() const
    { return this->getHighIndex(); }

    virtual void emitNormal(ostream& out, const IntVector& l, const IntVector& h,
			    ProblemSpecP /*varnode*/, bool outputDoubleAsFloat)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat())
	Array3<T>::write(out, l, h, outputDoubleAsFloat);
      else
	SCI_THROW(InternalError("Cannot yet write non-flat objects!\n", __FILE__, __LINE__));
    }

    virtual bool emitRLE(ostream& out, const IntVector& l, const IntVector& h,
			 ProblemSpecP /*varnode*/)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat()){
	SCIRun::RunLengthEncoder<T> rle(typename Array3<T>::iterator(this, l),
                                        typename Array3<T>::iterator(this, h));
	rle.write(out);
      }
      else
	SCI_THROW(InternalError("Cannot yet write non-flat objects!\n", __FILE__, __LINE__));
      return true;
    }

    virtual void readNormal(istream& in, bool swapBytes)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat())
	Array3<T>::read(in, swapBytes);
      else
	SCI_THROW(InternalError("Cannot yet read non-flat objects!\n", __FILE__, __LINE__));
    }
      
    virtual void readRLE(istream& in, bool swapBytes, int nByteMode)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat()){
	SCIRun::RunLengthEncoder<T> rle(in, swapBytes, nByteMode);
	rle.copyOut(Array3<T>::begin(), Array3<T>::end());
      }
      else
	SCI_THROW(InternalError("Cannot yet write non-flat objects!\n", __FILE__, __LINE__));
    }

    virtual RefCounted* getRefCounted() {
      return this->getWindow();
    }
  protected:
    GridVariable(const GridVariable<T>& copy) : Array3<T>(copy) {}
   
  private:
    GridVariable(Array3Window<T>* window)
      : Array3<T>(window) {}
    GridVariable<T>& operator=(const GridVariable<T>&);
  };

template<class T>
  void
  GridVariable<T>::copyPointer(Variable& copy)
  {
    GridVariable<T>* c = dynamic_cast<GridVariable<T>* >(&copy);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in Grid variable", __FILE__, __LINE__));
    copyPointer(*c);
  }

  template<class T>
  const GridVariable<T>& GridVariable<T>::castFromBase(const GridVariableBase* srcptr)
  {
    const GridVariable<T>* c = dynamic_cast<const GridVariable<T>* >(srcptr);
    if(!c)
      SCI_THROW(TypeMismatchException("Type mismatch in CC variable", __FILE__, __LINE__));
    return *c;
  }

  template<class T>
  void GridVariable<T>::allocate(const IntVector& lowIndex,
			       const IntVector& highIndex)
  {
    if(this->getWindow())
      SCI_THROW(InternalError("Allocating a Gridvariable that "
			  "is apparently already allocated!", __FILE__, __LINE__));
    this->resize(lowIndex, highIndex);
  }

  template<class T>
  void
  GridVariable<T>::copyPatch(const GridVariable<T>& src,
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
  GridVariable<T>::getSizes(IntVector& low, IntVector& high, 
			  IntVector& siz) const
  {
    low = this->getLowIndex();
    high = this->getHighIndex();
    siz = this->size();
  }

  template<class T>
  void
  GridVariable<T>::getSizes(IntVector& low, IntVector& high,
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
