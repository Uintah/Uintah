
#ifndef UINTAH_HOMEBREW_Reductions_H
#define UINTAH_HOMEBREW_Reductions_H

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
      
       Copyright (C) 2000 SCI Group
      
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
           fun_getLargeValue(&tmp);
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

