#ifndef UINTAH_HOMEBREW_fixedvector_H
#define UINTAH_HOMEBREW_fixedvector_H

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

      fixedvector() {
	 curalloc=Len;
	 cursize=0;
	 data=&fixed[0];
      }
      ~fixedvector() {
	 if(data != &fixed[0])
	    delete[] data;
      }

      iterator begin() { return &data[0]; }
      const_iterator begin() const { return &data[0]; }
      iterator end() { return &data[cursize]; }
      const_iterator end() const { return &data[cursize]; }

      void push_back(const T& x) {
	 if(cursize>=curalloc){
	    enlarge();
	 }
	 data[cursize++]=x;
      }


      void enlarge() {
	 curalloc+=(curalloc>>1);
	 T* newdata = new T[curalloc];
	 for(int i=0;i<cursize;i++)
	    newdata[i]=data[i];
	 if(data != &fixed[0])
	    delete[] data;
	 data=newdata;
	 //cerr << "dynamic, size=" << curalloc << '\n';
      }

      size_type size() const { return cursize; }
      void resize(size_type newsize) { cursize = newsize; }
      reference operator[](size_type n) { return data[n]; }
      const_reference operator[](size_type n) const {return data[n]; }
   private:
      fixedvector(const fixedvector<T,Len>& copy);
      fixedvector<T,Len>& operator=(const fixedvector<T,Len>& copy);
     
      T* data;
      T fixed[Len];
      int cursize;
      int curalloc;
   };

} // End namespace Uintah

#endif
