#ifndef UINTAH_HOMEBREW_Array3Window_H
#define UINTAH_HOMEBREW_Array3Window_H

#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Packages/Uintah/Core/Grid/Array3Data.h>
#include <limits.h>
#include <Core/Geometry/IntVector.h>

/**************************************

CLASS
   Array3Window
   
GENERAL INFORMATION

   Array3Window.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Array3Window

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

namespace Uintah {
   template<class T> class Array3Window : public RefCounted {
   public:
      Array3Window(Array3Data<T>*);
      Array3Window(Array3Data<T>*, const IntVector& offset,
		   const IntVector& lowIndex, const IntVector& highIndex);
      virtual ~Array3Window();
      
      inline const Array3Data<T>* getData() const {
	 return data;
      }
     
      inline Array3Data<T>* getData() {
	 return data;
      }
      
      void copy(const Array3Window<T>*);
      void copy(const Array3Window<T>*, const IntVector& low, const IntVector& high);
      void initialize(const T&);
      void initialize(const T&, const IntVector& s, const IntVector& e);
      inline IntVector getLowIndex() const {
	 return lowIndex;
      }
      inline IntVector getHighIndex() const {
	 return highIndex;
      }
      inline IntVector getOffset() const {
	 return offset;
      }
      inline T& get(const IntVector& idx) {
	 ASSERT(data);
	 CHECKARRAYBOUNDS(idx.x(), lowIndex.x(), highIndex.x());
	 CHECKARRAYBOUNDS(idx.y(), lowIndex.y(), highIndex.y());
	 CHECKARRAYBOUNDS(idx.z(), lowIndex.z(), highIndex.z());
	 return data->get(idx-offset);
      }
      
      ///////////////////////////////////////////////////////////////////////
      // Return pointer to the data 
      // (**WARNING**not complete implementation)
      inline T* getPointer() {
	return data ? (data->getPointer()) : 0;
      }
      
      ///////////////////////////////////////////////////////////////////////
      // Return const pointer to the data 
      // (**WARNING**not complete implementation)
      inline const T* getPointer() const {
	return (data->getPointer());
      }

      inline T*** get3DPointer() {
	return data ? data->get3DPointer():0;
      }
      inline T*** get3DPointer() const {
	return data ? data->get3DPointer():0;
      }

   private:
      
      Array3Data<T>* data;
      IntVector offset;
      IntVector lowIndex;
      IntVector highIndex;
      Array3Window(const Array3Window<T>&);
      Array3Window<T>& operator=(const Array3Window<T>&);
   };
   
   template<class T>
      void Array3Window<T>::initialize(const T& val)
      {
	 data->initialize(val, lowIndex-offset, highIndex-offset);
      }
   
   template<class T>
      void Array3Window<T>::copy(const Array3Window<T>* from)
      {
	 data->copy(lowIndex-offset, highIndex-offset, from->data,
		    from->lowIndex-from->offset, from->highIndex-from->offset);
      }
   
   template<class T>
      void Array3Window<T>::copy(const Array3Window<T>* from,
				 const IntVector& low, const IntVector& high)
      {
	 data->copy(low-offset, high-offset, from->data,
		    low-from->offset, high-from->offset);
      }
   
   template<class T>
      void Array3Window<T>::initialize(const T& val,
				       const IntVector& s,
				       const IntVector& e)
      {
	 CHECKARRAYBOUNDS(s.x(), lowIndex.x(), highIndex.x());
	 CHECKARRAYBOUNDS(s.y(), lowIndex.y(), highIndex.y());
	 CHECKARRAYBOUNDS(s.z(), lowIndex.z(), highIndex.z());
	 CHECKARRAYBOUNDS(e.x(), s.x(), highIndex.x()+1);
	 CHECKARRAYBOUNDS(e.y(), s.y(), highIndex.y()+1);
	 CHECKARRAYBOUNDS(e.z(), s.z(), highIndex.z()+1);
	 data->initialize(val, s-offset, e-offset);
      }
   
   template<class T>
      Array3Window<T>::Array3Window(Array3Data<T>* data)
      : data(data), offset(0,0,0), lowIndex(0,0,0), highIndex(data->size())
      {
	 data->addReference();
      }
   
   template<class T>
      Array3Window<T>::Array3Window(Array3Data<T>* data,
				    const IntVector& offset,
				    const IntVector& lowIndex,
				    const IntVector& highIndex)
      : data(data), offset(offset), lowIndex(lowIndex), highIndex(highIndex)
      {
	// null data can be used for a place holder in OnDemandDataWarehouse
	if (data != 0) {
	  CHECKARRAYBOUNDS(lowIndex.x()-offset.x(), 0, data->size().x());
	  CHECKARRAYBOUNDS(lowIndex.y()-offset.y(), 0, data->size().y());
	  CHECKARRAYBOUNDS(lowIndex.z()-offset.z(), 0, data->size().z());
	  CHECKARRAYBOUNDS(highIndex.x()-offset.x(), 0, data->size().x()+1);
	  CHECKARRAYBOUNDS(highIndex.y()-offset.y(), 0, data->size().y()+1);
	  CHECKARRAYBOUNDS(highIndex.z()-offset.z(), 0, data->size().z()+1);
	  data->addReference();
	}
	else {
	  // To use null data, put the offset as {INT_MAX, INT_MAX, INT_MAX}.
	  // This way, when null is used accidentally, this assertion will
	  // fail while allowing purposeful (and hopefully careful) uses
	  // of null data.
	  ASSERT(offset == IntVector(INT_MAX, INT_MAX, INT_MAX));
	}
      }
   
   template<class T>
      Array3Window<T>::~Array3Window()
      {
	 if(data && data->removeReference())
	    delete data;
      }
} // End namespace Uintah

#endif
