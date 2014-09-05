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
 *  CenterConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Center_Constraint_h
#define SCI_project_Center_Constraint_h 1

#include <Dataflow/share/share.h>
#include <Dataflow/Constraints/BaseConstraint.h>

namespace SCIRun {

// This constraint only finds the center (i.e. one-way constraint).

class PSECORESHARE CenterConstraint : public BaseConstraint {
public:
   CenterConstraint( const string& name,
		     const Index numSchemes,
		     PointVariable* center,
		     PointVariable* p1, PointVariable* p2 );
   CenterConstraint( const string& name,
		     const Index numSchemes,
		     PointVariable* center,
		     PointVariable* p1, PointVariable* p2,
		     PointVariable* p3 );
   CenterConstraint( const string& name,
		     const Index numSchemes,
		     PointVariable* center,
		     PointVariable* p1, PointVariable* p2,
		     PointVariable* p3, PointVariable* p4 );
   virtual ~CenterConstraint();
   
protected:
   virtual bool Satisfy( const Index index, const Scheme scheme,
			 const double Epsilon,
			 BaseVariable*& var, VarCore& c );
};

} // End namespace SCIRun


#endif
