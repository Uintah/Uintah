#ifndef UINTAH_HOMEBREW_SFCXVARIABLE_H
#define UINTAH_HOMEBREW_SFCXVARIABLE_H

#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/SFCXVariableBase.h>
#include <Packages/Uintah/Core/Grid/constGridVariable.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
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
    void fillFace(const Patch* patch, Patch::FaceType face,
                  const T& value, IntVector offset = IntVector(0,0,0))
    { 
      //__________________________________
      // Add (1,0,0) to low index when no 
      // neighbor patches are present
      IntVector low,hi; 
      int numGC = 0;
      low = patch->getCellLowIndex();
      low+= IntVector(patch->getBCType(Patch::xminus)==Patch::Neighbor?numGC:1,
		       patch->getBCType(Patch::yminus)==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::zminus)==Patch::Neighbor?numGC:0);
      low-= offset;
      hi  = patch->getCellHighIndex();
      hi += IntVector(patch->getBCType(Patch::xplus) ==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::yplus) ==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::zplus) ==Patch::Neighbor?numGC:0);
      hi += offset;
      // cout<< "fillFace: SFCXVariable.h"<<endl;
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
      // Add (1,0,0) to low index when no 
      // neighbor patches are present
      IntVector low,hi;  
      int numGC = 0;
      low = patch->getCellLowIndex();
      low+=IntVector(patch->getBCType(Patch::xminus)==Patch::Neighbor?numGC:1,
		       patch->getBCType(Patch::yminus)==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::zminus)==Patch::Neighbor?numGC:0);
      low-= offset;
      hi  = patch->getCellHighIndex();
      hi +=IntVector(patch->getBCType(Patch::xplus) ==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::yplus) ==Patch::Neighbor?numGC:0,
		       patch->getBCType(Patch::zplus) ==Patch::Neighbor?numGC:0);
      hi += offset;
      // cout<< "fillFaceflux: SFCXVariable.h"<<endl;
      // cout<< "low: "<<low<<endl;
      // cout<< "hi:  "<<hi <<endl;     

      switch (face) {
      case Patch::xplus:
	for (int j = low.y(); j<hi.y(); j++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(hi.x()-1,j,k)] = 
	      (*this)[IntVector(hi.x()-2,j,k)] + value*dx.x();
	  }
	}
	break;
      case Patch::xminus:
	for (int j = low.y(); j<hi.y(); j++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(low.x(),j,k)] = 
	      (*this)[IntVector(low.x()+1,j,k)] - value * dx.x();
	  }
	}
	break;
      case Patch::yplus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,hi.y()-1,k)] = 
	      (*this)[IntVector(i,hi.y()-2,k)] + value * dx.y();
	  }
	}
	break;
      case Patch::yminus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,low.y(),k)] = 
	      (*this)[IntVector(i,low.y()+1,k)] - value * dx.y();
	  }
	}
	break;
      case Patch::zplus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,hi.z()-1)] = 
	      (*this)[IntVector(i,j,hi.z()-2)] + value * dx.z();
	  }
	}
	break;
      case Patch::zminus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,low.z())] = 
	      (*this)[IntVector(i,j,low.z()+1)] -  value * dx.z();
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
			IntVector offset = IntVector(0,0,0))
    {
      IntVector low,hi;
      low = getLowIndex() + offset;
      hi = getHighIndex() - offset;
      switch (face) {
      case Patch::xplus:
	for (int j = low.y(); j<hi.y(); j++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(hi.x()-1,j,k)] =
	      Vector(0.0,(*this)[IntVector(hi.x()-1,j,k)].y(),
		     (*this)[IntVector(hi.x()-1,j,k)].z());
	  }
	}
	break;
      case Patch::xminus:
	for (int j = low.y(); j<hi.y(); j++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(low.x(),j,k)] = 
	      Vector(0.0,(*this)[IntVector(low.x(),j,k)].y(),
		     (*this)[IntVector(low.x(),j,k)].z());
	  }
	}
	break;
      case Patch::yplus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,hi.y()-1,k)] =
	      Vector((*this)[IntVector(i,hi.y()-1,k)].x(),0.0,
		     (*this)[IntVector(i,hi.y()-1,k)].z());
	  }
	}
	break;
      case Patch::yminus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,low.y(),k)] =
	      Vector((*this)[IntVector(i,low.y(),k)].x(),0.0,
		     (*this)[IntVector(i,low.y(),k)].z());
	  }
	}
	break;
      case Patch::zplus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,hi.z()-1)] =
	      Vector((*this)[IntVector(i,j,hi.z()-1)].x(),
		     (*this)[IntVector(i,j,hi.z()-1)].y(),0.0);
	  }
	}
	break;
      case Patch::zminus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,low.z())] =
	      Vector((*this)[IntVector(i,j,low.z())].x(),
		     (*this)[IntVector(i,j,low.z())].y(),0.0);
	  }
	}
	break;
      case Patch::invalidFace:
	break;
      }
    };
     
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
