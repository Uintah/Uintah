#ifndef __CONTACT_MATERIAL_SPEC_H__
#define __CONTACT_MATERIAL_SPEC_H__

#include <Core/Containers/StaticArray.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>

#include <vector>

namespace Uintah {
using namespace SCIRun;

/**************************************

CLASS
   ContactMaterialSpec
   
GENERAL INFORMATION

   ContactMaterialPairs.h

   Andrew Brydon
   Los Alamos National Laboratory
 
   Copyright (C) 2005 SCI Group

KEYWORDS
   Contact_Model Composite

DESCRIPTION
   allows contact conditions to be applied between 
   restricted set of materials.
  
   The input to this lives inside the contact block; 
   typical usage would look like

   <contact>
      <type>single_velocity</type>
      <materials>[0,1]</materials>
   </contact>

   which applies a single velocity contact between materials 0 and 1
   ignoring all cells that do not contain both materials.

   with more materials, such as 
   <contact>
      <type>rigid</type>
      <materials>[0,1,2]</materials>
   </contact>
   all the materials must be present for the contact condition to be 
   imposed. I guess you could use this for corner conditions.


WARNING

****************************************/

    class ContactMaterialSpec {
      
      public:
         // Constructor
         ContactMaterialSpec() {}
         
         // contructor using contact block
         ContactMaterialSpec(ProblemSpecP & ps);
         
         // require this material to apply contact
         void add(unsigned int matlIndex);
         
         // is this material used
         bool requested(int imat) const { 
           if(d_matls.size()==0)    return true; // everything by default
           if((int)d_matls.size()<=imat) return false;
           return d_matls[imat]; 
         }
          
         //  does this cell have the requested materials
         bool present(const StaticArray<constNCVariable<double> > & gmass, IntVector c) const
         {
           static const double EPSILON=1.e-12;
           
           size_t numMats = gmass.size();
           if(numMats>d_matls.size()) numMats = d_matls.size();
           
           for(unsigned int imat=0;imat<numMats;imat++) 
             {
               if(d_matls[imat] && fabs(gmass[imat][c])<EPSILON ) {
                 return false; // required material not present, dont apply this bc
               }
             }
           
           return true;
         }
         
      private: // hide
         ContactMaterialSpec(const ContactMaterialSpec &);
         ContactMaterialSpec& operator=(const ContactMaterialSpec &);
         
      protected: // data
         // is each material required
         std::vector<bool> d_matls;
      };
      
} // End namespace Uintah

#endif // __COMPOSITE_CONTACT_H__
