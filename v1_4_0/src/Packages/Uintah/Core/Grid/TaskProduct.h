#ifndef UINTAH_HOMEBREW_TaskProduct_H
#define UINTAH_HOMEBREW_TaskProduct_H

#include <Packages/Uintah/Core/Grid/VarLabel.h>

namespace Uintah {
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
      TaskProduct(const Patch* patch, int matlIndex, const VarLabel* label)
	 : d_patch(patch), d_label(label), d_matlIndex(matlIndex){
      }
      TaskProduct(const TaskProduct& copy)
	 : d_patch(copy.d_patch), d_label(copy.d_label),
	   d_matlIndex(copy.d_matlIndex) {
      }
      
      ~TaskProduct() {
      }

      const Patch* getPatch() const {
	 return d_patch;
      }
      const VarLabel* getLabel() const {
	 return d_label;
      }

      bool operator<(const TaskProduct& p) const {
	 if(d_patch == p.d_patch) {
	    if(d_matlIndex == p.d_matlIndex){
	       VarLabel::Compare c;
	       return c(d_label, p.d_label);
	    } else {
	       return d_matlIndex < p.d_matlIndex;
	    }
	 } else {
	    if(d_matlIndex == p.d_matlIndex)
	       return d_patch < p.d_patch;
	    else
	       return d_matlIndex < p.d_matlIndex;
	 }
      }

   private:
      const Patch* d_patch;
      const VarLabel* d_label;
      int d_matlIndex;

      TaskProduct& operator=(const TaskProduct&);
      
   };

} // End namespace Uintah

#endif
