#ifndef UINTAH_HOMEBREW_fixedvector_H
#define UINTAH_HOMEBREW_fixedvector_H

#include <SCICore/Exceptions/InternalError.h>

namespace Uintah {
   
   /**************************************
     
     CLASS
       fixedvector
      
       Short Description...
      
     GENERAL INFORMATION
      
       fixedvector.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       fixedvector
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
   
   template<class T, int Len>
   class fixedvector {
   public:
      typedef T value_type;
      typedef value_type* pointer;
      typedef const value_type* const_pointer;
      typedef value_type* iterator;
      typedef const value_type* const_iterator;
      typedef value_type& reference;
      typedef const value_type& const_reference;
      typedef int size_type;
      typedef ptrdiff_t difference_type;

      fixedvector() {cursize=0;}
      ~fixedvector() {}
      fixedvector(const fixedvector<T,Len>&);
      fixedvector<T,Len>& operator=(const fixedvector<T,Len>&);

      iterator begin() { return &data[0]; }
      const_iterator begin() const { return &data[0]; }
      iterator end() { return &data[cursize]; }
      const_iterator end() const { return &data[cursize]; }

      void push_back(const T& x) {
	 if(cursize>=Len)
	    throw SCICore::Exceptions::InternalError("fixed vector too small");
	 data[cursize++]=x;
      }

      size_type size() const { return cursize; }
      void resize(size_type newsize) { cursize = newsize; }
      reference operator[](size_type n) { return data[n]; }
      const_reference operator[](size_type n) const {return data[n]; }
   private:
      T data[Len];
      int cursize;
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.1.2.1  2000/10/10 05:32:29  sparker
// fixedvector is a fixed-size version of std::vector
// SimpleString is a low-overhead string class
//
//

#endif

