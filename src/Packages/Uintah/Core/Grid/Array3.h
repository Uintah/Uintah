#ifndef UINTAH_HOMEBREW_ARRAY3_H
#define UINTAH_HOMEBREW_ARRAY3_H

#include <Packages/Uintah/Core/Grid/Array3Window.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>

namespace Uintah {

/**************************************

CLASS
   Array3
   
GENERAL INFORMATION

   Array3.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Array3

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

template<class T> class Array3 {
      
   public:
      Array3() {
	 d_window = 0;
      }
      Array3(int size1, int size2, int size3) {
	 d_window=scinew Array3Window<T>(new Array3Data<T>( IntVector(size1, size2, size3) ));
	 d_window->addReference();
      }
      Array3(const IntVector& lowIndex, const IntVector& highIndex) {
	 d_window = 0;
	 resize(lowIndex,highIndex);
      }
      virtual ~Array3();
      Array3(const Array3& copy)
	 : d_window(copy.d_window)
      {
	 if(d_window)
	    d_window->addReference();
      }
      
      Array3& operator=(const Array3& copy) {
	 if(copy.d_window)
	    copy.d_window->addReference();
	 if(d_window && d_window->removeReference()){
	    delete d_window;
	 }
	 d_window = copy.d_window;
	 return *this;
      }

      IntVector size() const {
	 return d_window->getData()->size();
      }
      void initialize(const T& value) {
	 d_window->initialize(value);
      }
      void copy(const Array3<T>& from) {
	 ASSERT(d_window != 0);
	 ASSERT(from.d_window != 0);
	 d_window->copy(from.d_window);
      }
      
      void copy(const Array3<T>& from, const IntVector& low, const IntVector& high) {
	 ASSERT(d_window != 0);
	 ASSERT(from.d_window != 0);
	 d_window->copy(from.d_window, low, high);
      }
      
      void initialize(const T& value, const IntVector& s,
		      const IntVector& e) {
	 d_window->initialize(value, s, e);
      }
      
      void resize(const IntVector& lowIndex, const IntVector& highIndex) {
	 if(d_window && d_window->removeReference())
	    delete d_window;
	 IntVector size = highIndex-lowIndex;
	 d_window=scinew Array3Window<T>(new Array3Data<T>(size), lowIndex, lowIndex, highIndex);
	 d_window->addReference();
      }
      inline T& operator[](const IntVector& idx) const {
	 return d_window->get(idx);
      }
      
      inline Array3Window<T>* getWindow() const {
	 return d_window;
      }
      inline T& operator[](const IntVector& idx) {
	 return d_window->get(idx);
      }
      
      IntVector getLowIndex() const {
	 return d_window->getLowIndex();
      }

      IntVector getHighIndex() const {
	return d_window->getHighIndex();
      }

      ///////////////////////////////////////////////////////////////////////
      // Get low index for fortran calls
      inline IntVector getFortLowIndex() {
	return d_window->getLowIndex();
      }
      
      ///////////////////////////////////////////////////////////////////////
      // Get high index for fortran calls
      inline IntVector getFortHighIndex() {
	return d_window->getHighIndex()-IntVector(1,1,1);
      }
      
      ///////////////////////////////////////////////////////////////////////
      // Return pointer to the data 
      // (**WARNING**not complete implementation)
      inline T* getPointer() {
	return (d_window->getPointer());
      }
      
      ///////////////////////////////////////////////////////////////////////
      // Return const pointer to the data 
      // (**WARNING**not complete implementation)
      inline const T* getPointer() const {
	return (d_window->getPointer());
      }

      void print(std::ostream& out) const {
	 IntVector l = d_window->getLowIndex();
	 IntVector h = d_window->getHighIndex();
	 out << "Variable from " << l << " to " << h << '\n';
	 for (int ii = l.x(); ii < h.x(); ii++) {
	    out << "variable for ii = " << ii << endl;
	    for (int jj = l.y(); jj < h.y(); jj++) {
	       for (int kk = l.z(); kk < h.z(); kk++) {
		  out.width(10);
		  out << (*this)[IntVector(ii,jj,kk)] << " " ; 
	       }
	       out << endl;
	    }
	 }
      }

	 
   private:
      Array3Window<T>* d_window;
   };
   
   template<class T>
      Array3<T>::~Array3()
      {
	 if(d_window && d_window->removeReference()){
	    delete d_window;
	 }
      }

} // End namespace Uintah

#endif
