
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
	 T operator()(T a, T b) {
	    return a<b?a:b;
	 }
      };

      template<class T> class Sum {
      public:
	 T operator()(T a, T b) {
	    return a+b;
	 }
      };

   private:
      Reductions(const Reductions&);
      Reductions& operator=(const Reductions&);

#ifdef __GNUG__      
   public:
       // Keep gcc quiet, to avoid:
       // warning: all member functions in class `Uintah::Reductions' are private
       Reductions(Reductions*);
#endif
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/09/25 20:37:43  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.2  2000/05/31 20:25:32  guilkey
// Added the beginnings of a Sum reduction, which would take data from
// multiple patches, materials, etc. and add them together.  The immediate
// application is for computing the strain energy and storing it.  I'm
// going to need some help with this.
//
// Revision 1.1  2000/05/02 06:07:22  sparker
// Implemented more of DataWarehouse and SerialMPM
//
//

#endif

