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
#include <iostream>

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
    //////////
    // Insert Documentation Here:
    virtual CCVariableBase* clone();
    virtual const CCVariableBase* clone() const;
    virtual CCVariableBase* cloneType() const
    { return scinew CCVariable<T>(); }
    virtual constCCVariableBase* cloneConstType() const
    { return scinew constGridVariable<CCVariableBase, CCVariable<T>, T>(); }

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

    // Replace the values on the indicated face with value
    void fillFace(Patch::FaceType face, const T& value, 
		  IntVector offset = IntVector(0,0,0) )
    { 
      IntVector low,hi;
      low = getLowIndex() + offset;
      hi = getHighIndex() - offset;
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
     
   // Replace the values on the indicated face with value
   // using a 1st order difference formula for a Neumann BC condition
   // The plus_minus_one variable allows for negative interior BC, which is
   // simply the (-1)* interior value.

   void fillFaceFlux(Patch::FaceType face, const T& value,const Vector& dx,
                  const double& plus_minus_one=1.0,
		    IntVector offset = IntVector(0,0,0)  )
    { 
      IntVector low,hi;
      low = getLowIndex() + offset;
      hi = getHighIndex() - offset;

      switch (face) {
      case Patch::xplus:
	 for (int j = low.y(); j<hi.y(); j++) {
	   for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(hi.x()-1,j,k)] = 
	      ((*this)[IntVector(hi.x()-2,j,k)])*plus_minus_one - 
             value*dx.x();
	   }
	 }
	 break;
      case Patch::xminus:
	 for (int j = low.y(); j<hi.y(); j++) {
	   for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(low.x(),j,k)] = 
	      ((*this)[IntVector(low.x()+1,j,k)])*plus_minus_one - 
             value * dx.x();
	   }
	 }
	 break;
      case Patch::yplus:
	 for (int i = low.x(); i<hi.x(); i++) {
	   for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,hi.y()-1,k)] = 
	      ((*this)[IntVector(i,hi.y()-2,k)])*plus_minus_one - 
             value * dx.y();
	   }
	 }
	 break;
      case Patch::yminus:
	 for (int i = low.x(); i<hi.x(); i++) {
	   for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,low.y(),k)] = 
	      ((*this)[IntVector(i,low.y()+1,k)])*plus_minus_one - 
             value * dx.y();
	   }
	 }
	 break;
      case Patch::zplus:
	 for (int i = low.x(); i<hi.x(); i++) {
	   for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,hi.z()-1)] = 
	      ((*this)[IntVector(i,j,hi.z()-2)])*plus_minus_one - 
             value * dx.z();
	   }
	 }
	 break;
      case Patch::zminus:
	 for (int i = low.x(); i<hi.x(); i++) {
	   for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,low.z())] =
	      ((*this)[IntVector(i,j,low.z()+1)])*plus_minus_one - 
             value * dx.z();
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
      case Patch::numFaces:
	break;
      case Patch::invalidFace:
	break;
      }
    };
    //______________________________________________________________________
    // Update pressure boundary conditions due to hydrostatic pressure
    // Note (*this) = pressure
    void setHydrostaticPressureBC(Patch::FaceType face, Vector& gravity,
                                const CCVariable<double>& rho,
                                const Vector& dx,
			           IntVector offset = IntVector(0,0,0))
     { 
	IntVector low,hi;
	low = getLowIndex() + offset;
	hi = getHighIndex() - offset;

      // cout<< "CCVARIABLE LO" << low <<endl;
      // cout<< "CCVARIABLE HI" << hi <<endl;

	switch (face) {
	case Patch::xplus:
	  for (int j = low.y(); j<hi.y(); j++) {
	    for (int k = low.z(); k<hi.z(); k++) {
	     (*this)[IntVector(hi.x()-1,j,k)] = 
		(*this)[IntVector(hi.x()-2,j,k)] + 
              gravity.x() * rho[IntVector(hi.x()-2,j,k)] * dx.x();
	    }
	  }
	  break;
	case Patch::xminus:
	  for (int j = low.y(); j<hi.y(); j++) {
	    for (int k = low.z(); k<hi.z(); k++) {
	     (*this)[IntVector(low.x(),j,k)] = 
		(*this)[IntVector(low.x()+1,j,k)] - 
              gravity.x() * rho[IntVector(low.x()+1,j,k)] * dx.x();;
	    }
	  }
	  break;
	case Patch::yplus:
	  for (int i = low.x(); i<hi.x(); i++) {
	    for (int k = low.z(); k<hi.z(); k++) {
	     (*this)[IntVector(i,hi.y()-1,k)] = 
		(*this)[IntVector(i,hi.y()-2,k)] + 
              gravity.y() * rho[IntVector(i,hi.y()-2,k)] * dx.y();
	    }
	  }
	  break;
	case Patch::yminus:
	  for (int i = low.x(); i<hi.x(); i++) {
	    for (int k = low.z(); k<hi.z(); k++) {
	     (*this)[IntVector(i,low.y(),k)] = 
		(*this)[IntVector(i,low.y()+1,k)] - 
              gravity.y() * rho[IntVector(i,low.y()+1,k)] * dx.y();
	    }
	  }
	  break;
	case Patch::zplus:
	  for (int i = low.x(); i<hi.x(); i++) {
	    for (int j = low.y(); j<hi.y(); j++) {
	     (*this)[IntVector(i,j,hi.z()-1)] = 
		(*this)[IntVector(i,j,hi.z()-2)] +
              gravity.z() * rho[IntVector(i,j,hi.z()-2)] * dx.z();
	    }
	  }
	  break;
	case Patch::zminus:
	  for (int i = low.x(); i<hi.x(); i++) {
	    for (int j = low.y(); j<hi.y(); j++) {
	     (*this)[IntVector(i,j,low.z())] =
		(*this)[IntVector(i,j,low.z()+1)] -  
              gravity.z() * rho[IntVector(i,j,low.z()+1)] * dx.z();
	    }
	  }
	  break;
	case Patch::numFaces:
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
    CCVariable(const CCVariable<T>&);

  private:
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
      throw TypeMismatchException("Type mismatch in CC variable");
    copyPointer(*c);
  }

 
  template<class T>
  CCVariable<T>::CCVariable()
  {
    //	 std::cerr << "CCVariable ctor not done!\n";
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
      throw InternalError("Allocating a CCvariable that "
			  "is apparently already allocated!");
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
      throw TypeMismatchException("Type mismatch in CC variable");
    return *c;
  }

  template<class T>
  void
  CCVariable<T>::copyPatch(const CCVariable<T>& src,
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

