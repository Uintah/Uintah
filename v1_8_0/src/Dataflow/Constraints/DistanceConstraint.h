/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  DistanceConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Distance_Constraint_h
#define SCI_project_Distance_Constraint_h 1

#include <Dataflow/share/share.h>
#include <Dataflow/Constraints/BaseConstraint.h>

namespace SCIRun {

class PSECORESHARE DistanceConstraint : public BaseConstraint {
public:
   DistanceConstraint( const string& name,
		       const Index numSchemes,
		       PointVariable* p1, PointVariable* p2,
		       RealVariable* dist );
   virtual ~DistanceConstraint();

   // Use this to set the default direction used when p1==p2.
   // Defaults to (1,0,0).
   void SetDefault( const Vector& v );
   void SetMinimum( const double min );
   
protected:
   virtual bool Satisfy( const Index index, const Scheme scheme,
			 const double Epsilon,
			 BaseVariable*& var, VarCore& c );

private:
   Vector guess;
   double minimum;
};

} // End namespace SCIRun


#endif
