/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef UINTAH_HOMEBREW_Reductions_H
#define UINTAH_HOMEBREW_Reductions_H

#include <Core/Disclosure/TypeUtils.h>

namespace Uintah {
   /**************************************
     
     CLASS
       Reductions
      
       Short Description...
      
     GENERAL INFORMATION
      
       Reductions.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
     KEYWORDS
       Reductions
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class Reductions {
   public:
      template<class T> class Min {
      public:
         T operator()(T a, T b) const {
            return a<b?a:b;
         }
         T getBenignValue() const {
           T tmp;
           fun_getLargeValue(&tmp);
           return tmp;
         }
      };

      public:
      template<class T> class Max {
      public:
         T operator()(T a, T b) const {
            return a>b?a:b;
         }
         T getBenignValue() const {
           T tmp;
           fun_getSmallValue(&tmp);
           return tmp;
         }
      };

      template<class T> class Sum {
      public:
         T operator()(T a, T b) const {
            return a+b;
         }
         T getBenignValue() const {
           T tmp;
           fun_getZeroValue(&tmp);
           return tmp;
         }
      };

      template<class T> class And {
      public:
        T operator()(T a, T b) const {
          return a && b;
         }
         T getBenignValue() const {
           T tmp;
           fun_getLargeValue(&tmp);
           return tmp;
         }
      };

   private:
      Reductions(const Reductions&);
      Reductions& operator=(const Reductions&);

#ifdef __GNUG__      
   public:
       // Keep gcc quiet, to avoid:
       // warning: all member functions in class `Packages/Uintah::Reductions' are private
       Reductions(Reductions*);
#endif
   };

} // End namespace Uintah

#endif

