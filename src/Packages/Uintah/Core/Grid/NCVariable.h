#ifndef UINTAH_HOMEBREW_NCVARIABLE_H
#define UINTAH_HOMEBREW_NCVARIABLE_H

#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/NCVariableBase.h>
#include <Packages/Uintah/Core/Grid/constGridVariable.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/CCA/Ports/InputContext.h>
#include <Packages/Uintah/CCA/Ports/OutputContext.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
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
    virtual void offsetGrid(IntVector offset)
    { Array3<T>::offset(offset); }    
    
    //////////
    // Insert Documentation Here:
    virtual NCVariableBase* clone();
    virtual const NCVariableBase* clone() const;    
    virtual NCVariableBase* cloneType() const
    { return scinew NCVariable<T>(); }
    virtual constNCVariableBase* cloneConstType() const
    { return scinew constGridVariable<NCVariableBase, NCVariable<T>, T>(); }

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
     
    // Replace the values on the indicated face with value
    void fillFace(const Patch* patch, Patch::FaceType face, const T& value, 
		  IntVector offset = IntVector(0,0,0))

   { 
   // cout <<"NCVariable.h: fillFace face "<<face<<endl;
      IntVector low,hi;
      low = patch->getInteriorNodeLowIndex();
      low+= offset;
    //cout <<"low "<<low-offset<<"  Low + offset "<<low<<endl;
                        
      hi = patch->getInteriorNodeHighIndex();      
      hi -= offset;
    //cout <<"high "<<hi+offset<<"  hi - offset "<<hi<<endl;
       
      switch (face) {
      case Patch::xplus:
	for (int j = low.y(); j<hi.y(); j++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(hi.x()-1,j,k)] = value;
           //cout<<"fillFace xPlus "<<"patch "<<patch->getID()<<" "<<
           //   IntVector(hi.x()-1,j,k)<<endl;
	  }
	}
	break;
      case Patch::xminus:
	for (int j = low.y(); j<hi.y(); j++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(low.x(),j,k)] = value;
           //cout<<"fillFace xMinus "<<"patch "<<patch->getID()<<" "<<
           //   IntVector(low.x(),j,k)<<endl;
	  }
	}
	break;
      case Patch::yplus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,hi.y()-1,k)] = value;
           //cout<<"fillFace yplus "<<"patch "<<patch->getID()<<" "<<
           //   IntVector(i,hi.y()-1,k)<<endl;
	  }
	}
	break;
      case Patch::yminus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int k = low.z(); k<hi.z(); k++) {
	    (*this)[IntVector(i,low.y(),k)] = value;
           //cout<<"fillFace yminus "<<"patch "<<patch->getID()<<" "<<
           //   IntVector(i,low.y(),k)<<endl;
	  }
	}
	break;
      case Patch::zplus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,hi.z()-1)] = value;
           //cout<<"fillFace zplus "<<"patch "<<patch->getID()<<" "<<
           //   IntVector(i,j,hi.z()-1)<<endl;
	  }
	}
	break;
      case Patch::zminus:
	for (int i = low.x(); i<hi.x(); i++) {
	  for (int j = low.y(); j<hi.y(); j++) {
	    (*this)[IntVector(i,j,low.z())] = value;
           //cout<<"fillFace zminus "<<"patch "<<patch->getID()<<" "<<
           //   IntVector(i,j,low.z())<<endl;
	  }
	}
	break;
      default:
	throw InternalError("Illegal FaceType in NCVariable::fillFace");
      }

    };
 
  
    // Use to apply symmetry boundary conditions.  On the
    // indicated face, replace the component of the vector
    // normal to the face with 0.0
    void fillFaceNormal(const Patch* patch, Patch::FaceType face, 
			IntVector offset = IntVector(0,0,0));
/*     { */
/*     //cout <<"NCVariable.h: fillFaceNormal face "<<face<<endl; */
/*       IntVector low,hi; */
/*       low = patch->getInteriorNodeLowIndex(); */
/*       low+= offset; */
/*     //cout <<"low "<<low-offset<<"  Low + offset "<<low<<endl; */
                        
/*       hi = patch->getInteriorNodeHighIndex();       */
/*       hi -= offset; */
/*     //cout <<"high "<<hi+offset<<"  hi - offset "<<hi<<endl; */

/*       switch (face) { */
/*       case Patch::xplus: */
/* 	for (int j = low.y(); j<hi.y(); j++) { */
/* 	  for (int k = low.z(); k<hi.z(); k++) { */
/* 	    (*this)[IntVector(hi.x()-1,j,k)] = */
/* 	      Vector(0.0,(*this)[IntVector(hi.x()-1,j,k)].y(), */
/* 		     (*this)[IntVector(hi.x()-1,j,k)].z()); */
/*             //cout<<"fillFaceFlux xPlus "<<"patch "<<patch->getID()<<" "<< */
/*             //    IntVector(hi.x()-1,j,k)<<endl; */
/* 	  } */
/* 	} */
/* 	break; */
/*       case Patch::xminus: */
/* 	for (int j = low.y(); j<hi.y(); j++) { */
/* 	  for (int k = low.z(); k<hi.z(); k++) { */
/* 	    (*this)[IntVector(low.x(),j,k)] =  */
/* 	      Vector(0.0,(*this)[IntVector(low.x(),j,k)].y(), */
/* 		     (*this)[IntVector(low.x(),j,k)].z()); */
/*            //cout<<"fillFaceFlux xMinus "<<"patch "<<patch->getID()<<" "<< */
/*            //    IntVector(low.x(),j,k)<<endl; */
/* 	  } */
/* 	} */
/* 	break; */
/*       case Patch::yplus: */
/* 	for (int i = low.x(); i<hi.x(); i++) { */
/* 	  for (int k = low.z(); k<hi.z(); k++) { */
/* 	    (*this)[IntVector(i,hi.y()-1,k)] = */
/* 	      Vector((*this)[IntVector(i,hi.y()-1,k)].x(),0.0, */
/* 		     (*this)[IntVector(i,hi.y()-1,k)].z()); */
/*            //cout<<"fillFaceFlux yplus "<<"patch "<<patch->getID()<<" "<< */
/*            //     IntVector(i,hi.y()-1,k)<<endl; */
/* 	  } */
/* 	} */
/* 	break; */
/*       case Patch::yminus: */
/* 	for (int i = low.x(); i<hi.x(); i++) { */
/* 	  for (int k = low.z(); k<hi.z(); k++) { */
/* 	    (*this)[IntVector(i,low.y(),k)] = */
/* 	      Vector((*this)[IntVector(i,low.y(),k)].x(),0.0, */
/* 		     (*this)[IntVector(i,low.y(),k)].z()); */
/*            //cout<<"fillFaceFlux yminus "<<"patch "<<patch->getID()<<" "<< */
/*            //     IntVector(i,low.y(),k)<<endl; */
/* 	  } */
/* 	} */
/* 	break; */
/*       case Patch::zplus: */
/* 	for (int i = low.x(); i<hi.x(); i++) { */
/* 	  for (int j = low.y(); j<hi.y(); j++) { */
/* 	    (*this)[IntVector(i,j,hi.z()-1)] = */
/* 	      Vector((*this)[IntVector(i,j,hi.z()-1)].x(), */
/* 		     (*this)[IntVector(i,j,hi.z()-1)].y(),0.0); */
/*            //cout<<"fillFaceFlux zplus "<<"patch "<<patch->getID()<<" "<< */
/*            //     IntVector(i,j,hi.z()-1)<<endl; */
/* 	  } */
/* 	} */
/* 	break; */
/*       case Patch::zminus: */
/* 	for (int i = low.x(); i<hi.x(); i++) { */
/* 	  for (int j = low.y(); j<hi.y(); j++) { */
/* 	    (*this)[IntVector(i,j,low.z())] = */
/* 	      Vector((*this)[IntVector(i,j,low.z())].x(), */
/* 		     (*this)[IntVector(i,j,low.z())].y(),0.0); */
/*            //cout<<"fillFace zminus "<<"patch "<<patch->getID()<<" "<< */
/*            //     IntVector(i,j,low.z())<<endl; */
/* 	  } */
/* 	} */
/* 	break; */
/*       default: */
/* 	throw InternalError("Illegal FaceType in NCVariable::fillFaceNormal"); */
/*       } */
/*     }; */
     
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
    NCVariable(const NCVariable<T>&);
  private:
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
   
  // Use to apply symmetry boundary conditions.  On the
  // indicated face, replace the component of the vector
  // normal to the face with 0.0
  template<>
    void
    NCVariable<Vector>::fillFaceNormal(const Patch* patch,
				       Patch::FaceType face, 
				       IntVector offset);

  template<class T>
    void
    NCVariable<T>::fillFaceNormal(const Patch*, Patch::FaceType, 
				  IntVector)
    {
      return;
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
      throw TypeMismatchException("Type mismatch in NC variable");
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
      throw InternalError("Allocating an NCvariable that "
			  "is apparently already allocated!");
    resize(lowIndex, highIndex);
  }

  template<class T>
  const NCVariable<T>& NCVariable<T>::castFromBase(const NCVariableBase* srcptr)
  {
    const NCVariable<T>* c = dynamic_cast<const NCVariable<T>* >(srcptr);
    if(!c)
      throw TypeMismatchException("Type mismatch in NC variable");
    return *c;
  }

  template<class T>
  void
  NCVariable<T>::copyPatch(const NCVariable<T>& src,
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
