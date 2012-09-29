/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_STENCIL_H
#define UINTAH_HOMEBREW_STENCIL_H

#include <Core/Grid/Variables/DataItem.h>
#include <Core/Exceptions/TypeMismatchException.h>

namespace Uintah {

class TypeDescription;

/**************************************

CLASS
   Stencil
   
   Short description...

GENERAL INFORMATION

   Stencil.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Variable__Cell_Centered

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   template<class T> class Stencil : public DataItem {
   public:
      Stencil();
      Stencil(const Patch*);
      Stencil(const Stencil<T>&);
      virtual ~Stencil();
      
      //////////
      // Insert Documentation Here:
      static const TypeDescription* getTypeDescription();
      
      //////////
      // Insert Documentation Here:
      virtual void get(DataItem&) const;
      
      //////////
      // Insert Documentation Here:
      virtual Stencil<T>* clone() const;
      
      //////////
      // Insert Documentation Here:
      virtual void allocate(const Patch*);
      
      Stencil<T>& operator=(const Stencil<T>&);
   private:
   };
   
   template<class T>
      const TypeDescription*
      Stencil<T>::getTypeDescription()
      {
         //cerr << "Stencil::getTypeDescription not done\n";
         return 0;
      }
   
   template<class T>
      Stencil<T>::~Stencil()
      {
      }
   
   template<class T>
      void
      Stencil<T>::get(DataItem& copy) const
      {
         Stencil<T>* ref=dynamic_cast<Stencil<T>*>(&copy);
         if(!ref)
           SCI_THROW(TypeMismatchException("Stencil<T>", __FILE__, __LINE__));
         *ref = *this;
      }
   
   template<class T>
      Stencil<T>*
      Stencil<T>::clone() const
      {
         return scinew Stencil<T>(*this);
      }
   
   template<class T>
      Stencil<T>&
      Stencil<T>::operator=(const Stencil<T>& copy)
      {
         if(this != &copy){
            std::cerr << "Stencil<T>::operator= not done!\n";
         }
         return *this;
      }
   
   template<class T>
      Stencil<T>::Stencil()
      {
         std::cerr << "Stencil ctor not done!\n";
      }
   
   template<class T>
      Stencil<T>::Stencil(const Stencil<T>& copy)
      {
         std::cerr << "Stencil copy ctor not done!\n";
      }
   
   template<class T>
      void Stencil<T>::allocate(const Patch*)
      {
         std::cerr << "Stencil::allocate not done!\n";
      }
} // End namespace Uintah

#endif
