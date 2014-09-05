/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <Dataflow/Constraints/BaseConstraint.h>

#include <Dataflow/Constraints/share.h>

namespace SCIRun {

// This constraint only finds the center (i.e. one-way constraint).

class SCISHARE CenterConstraint : public BaseConstraint {
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
