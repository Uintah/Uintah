#ifndef UINTAH_HOMEBREW_TaskProduct_H
#define UINTAH_HOMEBREW_TaskProduct_H

#include <Uintah/Grid/VarLabel.h>

namespace Uintah {
   class Region;
   
   /**************************************
     
     CLASS
       TaskProduct
      
       Short Description...
      
     GENERAL INFORMATION
      
       TaskProduct.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       TaskProduct
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class TaskProduct {
   public:
      TaskProduct(const Region* region, int matlIndex, const VarLabel* label)
	 : d_region(region), d_matlIndex(matlIndex), d_label(label){
      }
      TaskProduct(const TaskProduct& copy)
	 : d_region(copy.d_region), d_label(copy.d_label),
	   d_matlIndex(copy.d_matlIndex) {
      }
      
      ~TaskProduct() {
      }

      const Region* getRegion() const {
	 return d_region;
      }
      const VarLabel* getLabel() const {
	 return d_label;
      }

      bool operator<(const TaskProduct& p) const {
	 if(d_region == p.d_region) {
	    if(d_matlIndex == p.d_matlIndex){
	       VarLabel::Compare c;
	       return c(d_label, p.d_label);
	    } else {
	       return d_matlIndex < p.d_matlIndex;
	    }
	 } else {
	    if(d_matlIndex == p.d_matlIndex)
	       return d_region < p.d_region;
	    else
	       return d_matlIndex < p.d_matlIndex;
	 }
      }

   private:
      const Region* d_region;
      const VarLabel* d_label;
      int d_matlIndex;

      TaskProduct& operator=(const TaskProduct&);
      
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/05/07 06:02:13  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
//

#endif

