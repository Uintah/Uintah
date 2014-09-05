#ifndef UINTAH_HOMEBREW_SFCZVARIABLE_H
#define UINTAH_HOMEBREW_SFCZVARIABLE_H

#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/SFCZVariableBase.h>
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

    virtual void copyPointer(SFCZVariableBase&);

    virtual bool rewindow(const IntVector& low, const IntVector& high)
    { return Array3<T>::rewindow(low, high); }

    //////////
    // Insert Documentation Here:
    virtual SFCZVariableBase* clone();
    virtual const SFCZVariableBase* clone() const;    
    virtual SFCZVariableBase* cloneType() const
    { return scinew SFCZVariable<T>(); }
    virtual constSFCZVariableBase* cloneConstType() const
    { return scinew constGridVariable<SFCZVariableBase, SFCZVariable<T>, T>();
    }

    //////////
    // Insert Documentation Here:
    virtual void allocate(const IntVector& lowIndex,
			  const IntVector& highIndex);
     
    virtual void allocate(const Patch* patch)
    { allocate(patch->getSFCZLowIndex(), patch->getSFCZHighIndex()); }
    virtual void allocate(const SFCZVariable<T>& src)
    { allocate(src.getLowIndex(), src.getHighIndex()); }
    virtual void allocate(const SFCZVariableBase* src)
    { allocate(castFromBase(src)); }

    void copyPatch(const SFCZVariable<T>& src,
		   const IntVector& lowIndex, const IntVector& highIndex);
    virtual void copyPatch(const SFCZVariableBase* src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex)
    { copyPatch(castFromBase(src), lowIndex, highIndex); }
    
    void copyData(const SFCZVariable<T>& src)
    { copyPatch(src, src.getLowIndex(), src.getHighIndex()); }
    virtual void copyData(const SFCZVariableBase* src)
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
      IntVector siz = size();
      ostringstream str;
      str << siz.x() << "x" << siz.y() << "x" << siz.z();
      elems=str.str();
      totsize=siz.x()*siz.y()*siz.z()*sizeof(T);
      ptr = (void*)getPointer();
    }

    // Replace the values on the indicated face with value
    void fillFace(const Patch* patch, Patch::FaceType face,
                  const T& value, IntVector offset = IntVector(0,0,0))
    { 
      //__________________________________
      // Add (0,0,1) to low index when no 
      // neighbor patches are present
      IntVector low,hi; 
      int numGC = 0;
      low = patch->getCellLowIndex();
      low+=IntVector(patch->getBCType(Patch::xminus)==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::yminus)==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::zminus)==Patch::Neighbor?numGC:1);
      low-= offset;
      hi  = patch->getCellHighIndex();
      hi +=IntVector(patch->getBCType(Patch::xplus) ==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::yplus) ==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::zplus) ==Patch::Neighbor?numGC:0);
      hi += offset;
      // cout<< "fillFace: SFCZVariable.h"<<endl;
      // cout<< "low: "<<low<<endl;
      // cout<< "hi:  "<<hi <<endl;

      switch (face) {
      case Patch::xplus:
	for (int j = low.y(); j<hi.y(); j++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(hi.x()-1,j,k)] = value;
	  }
	}
	break;
      case Patch::xminus:
	for (int j = low.y(); j<hi.y(); j++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(low.x(),j,k)] = value;
	  }
	}
	break;
      case Patch::yplus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,hi.y()-1,k)] = value;
	  }
	}
	break;
      case Patch::yminus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,low.y(),k)] = value;
	  }
	}
	break;
      case Patch::zplus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,hi.z()-1)] = value;
	  }
	}
	break;
      case Patch::zminus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,low.z())] = value;
	  }
	}
	break;
      case Patch::numFaces:
	break;
      case Patch::invalidFace:
	break;
      }

    };

    // Set the Neumann BC condition using a 1st order approximation
    void fillFaceFlux(const Patch* patch, Patch::FaceType face, 
                      const T& value, const Vector& dx,
		        IntVector offset = IntVector(0,0,0))
    { 
      //__________________________________
      // Add (0,0,1) to low index when no 
      // neighbor patches are present 
      IntVector low,hi; 
      int numGC = 0;
      low = patch->getCellLowIndex();
      low+=IntVector(patch->getBCType(Patch::xminus)==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::yminus)==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::zminus)==Patch::Neighbor?numGC:1);
      low-= offset;
      hi  = patch->getCellHighIndex();
      hi +=IntVector(patch->getBCType(Patch::xplus) ==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::yplus) ==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::zplus) ==Patch::Neighbor?numGC:0);
      hi += offset;
      // cout<< "fillFaceflux: SFCZVariable.h"<<endl;
      // cout<< "low: "<<low<<endl;
      // cout<< "hi:  "<<hi <<endl;     

      switch (face) {
      case Patch::xplus:
	for (int j = low.y(); j<hi.y(); j++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(hi.x()-1,j,k)] = 
	      (T)((*this)[IntVector(hi.x()-2,j,k)] + value*dx.x());
	  }
	}
	break;
      case Patch::xminus:
	for (int j = low.y(); j<hi.y(); j++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(low.x(),j,k)] = 
	      (T)((*this)[IntVector(low.x()+1,j,k)] - value * dx.x());
	  }
	}
	break;
      case Patch::yplus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,hi.y()-1,k)] = 
	      (T)((*this)[IntVector(i,hi.y()-2,k)] + value * dx.y());
	  }
	}
	break;
      case Patch::yminus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,low.y(),k)] = 
	      (T)((*this)[IntVector(i,low.y()+1,k)] - value * dx.y());
	  }
	}
	break;
      case Patch::zplus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,hi.z()-1)] = 
	      (T)((*this)[IntVector(i,j,hi.z()-2)] + value * dx.z());
	  }
	}
	break;
      case Patch::zminus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,low.z())] =
	      (T)((*this)[IntVector(i,j,low.z()+1)] -  value * dx.z());
	  }
	}
	break;
      case Patch::numFaces:
	break;
      case Patch::invalidFace:
	break;
      }

    };
     
     
    // Use to apply symmetry boundary conditions.  On the
    // indicated face, replace the component of the vector
    // normal to the face with 0.0
    void fillFaceNormal(Patch::FaceType face,
			IntVector offset = IntVector(0,0,0));
     
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
    SFCZVariable(const SFCZVariable<T>&);
  private:
    SFCZVariable<T>& operator=(const SFCZVariable<T>&);

    static const SFCZVariable<T>& castFromBase(const SFCZVariableBase* srcptr);
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
   
  // Use to apply symmetry boundary conditions.  On the
  // indicated face, replace the component of the vector
  // normal to the face with 0.0
  template<>
  void
  SFCZVariable<Vector>::fillFaceNormal(Patch::FaceType face, 
				       IntVector offset);
  template<class T>
  void
  SFCZVariable<T>::fillFaceNormal(Patch::FaceType, IntVector)
  {
    return;
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
  SFCZVariableBase*
  SFCZVariable<T>::clone()
  {
    return scinew SFCZVariable<T>(*this);
  }

  template<class T>
  const SFCZVariableBase*
  SFCZVariable<T>::clone() const
  {
    return scinew SFCZVariable<T>(*this);
  }
   
  template<class T>
  void
  SFCZVariable<T>::copyPointer(SFCZVariableBase& copy)
  {
    SFCZVariable<T>* c = dynamic_cast<SFCZVariable<T>* >(&copy);
    if(!c)
      throw TypeMismatchException("Type mismatch in SFCZ variable");
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
    if(getWindow())
      throw InternalError("Allocating an SFCZvariable that "
			  "is apparently already allocated!");
    resize(lowIndex, highIndex);
  }
/*
  template<class T>
  void SFCZVariable<T>::rewindow(const IntVector& low,
				 const IntVector& high) {
    Array3<T> newdata;
    newdata.resize(low, high);
    newdata.copy(*this, low, high);
    resize(low, high);
    Array3<T>::operator=(newdata);
  }
*/

  template<class T>
  const SFCZVariable<T>& SFCZVariable<T>::castFromBase(const SFCZVariableBase* srcptr)
  {
    const SFCZVariable<T>* c = dynamic_cast<const SFCZVariable<T>* >(srcptr);
    if(!c)
      throw TypeMismatchException("Type mismatch in SFCZ variable");
    return *c;
  }

  template<class T>
  void
  SFCZVariable<T>::copyPatch(const SFCZVariable<T>& src,
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
  SFCZVariable<T>::getBasePointer() const
  {
    return (void*)getPointer();
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
    low = getLowIndex();
    high = getHighIndex();
    siz = size();
  }

  template<class T>
  void
  SFCZVariable<T>::getSizes(IntVector& low, IntVector& high,
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
