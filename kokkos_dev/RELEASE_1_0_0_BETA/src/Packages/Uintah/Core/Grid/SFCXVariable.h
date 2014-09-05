#ifndef UINTAH_HOMEBREW_SFCXVARIABLE_H
#define UINTAH_HOMEBREW_SFCXVARIABLE_H

#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/SFCXVariableBase.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Malloc/Allocator.h>
#include <unistd.h>
#include <errno.h>

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
  public:
     
    SFCXVariable();
    SFCXVariable(const SFCXVariable<T>&);
    virtual ~SFCXVariable();
     
    //////////
    // Insert Documentation Here:
    static const TypeDescription* getTypeDescription();
     
    virtual void rewindow(const IntVector& low, const IntVector& high);
    virtual void copyPointer(const SFCXVariableBase&);
     
    //////////
    // Insert Documentation Here:
    virtual SFCXVariableBase* clone() const;
     
    //////////
    // Insert Documentation Here:
    virtual void allocate(const IntVector& lowIndex,
			  const IntVector& highIndex);

    virtual void allocate(const Patch* patch)
    { allocate(patch->getSFCXLowIndex(), patch->getSFCXHighIndex()); }
      
    virtual void copyPatch(SFCXVariableBase* src,
			   const IntVector& lowIndex,
			   const IntVector& highIndex);
    SFCXVariable<T>& operator=(const SFCXVariable<T>&);
     
    virtual void* getBasePointer();
    virtual const TypeDescription* virtualGetTypeDescription() const;
    virtual void getSizes(IntVector& low, IntVector& high,
			  IntVector& siz) const;
    virtual void getSizes(IntVector& low, IntVector& high,
			  IntVector& siz, IntVector& strides) const;

    // Replace the values on the indicated face with value
    void fillFace(Patch::FaceType face, const T& value, 
		  IntVector offset = IntVector(0,0,0))
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

    // Set the Neumann BC condition using a 1st order approximation
    void fillFaceFlux(Patch::FaceType face, const T& value, const Vector& dx,
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
  
    virtual void emitRLE(ostream& out, DOM_Element /*varnode*/)
    {
      const TypeDescription* td = fun_getTypeDescription((T*)0);
      if(td->isFlat()){
	RunLengthEncoder<T> rle(Array3<T>::begin(), Array3<T>::end());
	rle.write(out);
      }
      else
	throw InternalError("Cannot yet write non-flat objects!\n");
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
  private:
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
  SFCXVariable<T>::clone() const
  {
    return scinew SFCXVariable<T>(*this);
  }
   
  template<class T>
  void SFCXVariable<T>::rewindow(const IntVector& low,
				 const IntVector& high) {
    Array3<T> newdata;
    newdata.resize(low, high);
    newdata.copy(*this, low, high);
    resize(low, high);
    Array3<T>::operator=(newdata);
  }
  template<class T>
  void
  SFCXVariable<T>::copyPointer(const SFCXVariableBase& copy)
  {
    const SFCXVariable<T>* c = dynamic_cast<const SFCXVariable<T>* >(&copy);
    if(!c)
      throw TypeMismatchException("Type mismatch in SFCX variable");
    *this = *c;
  }

  template<class T>
  SFCXVariable<T>&
  SFCXVariable<T>::operator=(const SFCXVariable<T>& copy)
  {
    if(this != &copy){
      Array3<T>::operator=(copy);
    }
    return *this;
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
  void
  SFCXVariable<T>::copyPatch(SFCXVariableBase* srcptr,
			     const IntVector& lowIndex,
			     const IntVector& highIndex)
  {
    const SFCXVariable<T>* c = dynamic_cast<const SFCXVariable<T>* >(srcptr);
    if(!c)
      throw TypeMismatchException("Type mismatch in SFCX variable");
    const SFCXVariable<T>& src = *c;
    for(int i=lowIndex.x();i<highIndex.x();i++)
      for(int j=lowIndex.y();j<highIndex.y();j++)
	for(int k=lowIndex.z();k<highIndex.z();k++)
	  (*this)[IntVector(i, j, k)] = src[IntVector(i,j,k)];
  }
   
  template<class T>
  void*
  SFCXVariable<T>::getBasePointer()
  {
    return getPointer();
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
  SFCXVariable<T>::getSizes(IntVector& low, IntVector& high, IntVector& siz,
			    IntVector& strides) const
  {
    low=getLowIndex();
    high=getHighIndex();
    siz=size();
    strides = IntVector(sizeof(T), (int)(sizeof(T)*siz.x()),
			(int)(sizeof(T)*siz.y()*siz.x()));
  }

} // end namespace Uintah

#endif
