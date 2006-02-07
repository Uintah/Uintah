/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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
 *  BaseConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <stdio.h>
#include <string.h>
#include <Dataflow/Constraints/BaseConstraint.h>
#include <iostream>
using std::endl;
using std::ostream;

namespace SCIRun {

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 * The BaseConstraint constructor allocates the constraint's standard
 *      structures.
 */
BaseConstraint::BaseConstraint( const string& name, const Index nschemes,
				const Index varCount )
: name(name), nschemes(nschemes), varCount(varCount),
  vars(varCount), var_indices(varCount), var_choices(nschemes, varCount)
{
   whichMethod = 0;
}


/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
BaseConstraint::~BaseConstraint()
{
}


void
BaseConstraint::Priorities( const VPriority p1,
			    const VPriority p2,
			    const VPriority p3,
			    const VPriority p4,
			    const VPriority p5,
			    const VPriority p6,
			    const VPriority p7 )
{
   Index p=0;
   
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indices[p], p1);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indices[p], p2);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indices[p], p3);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indices[p], p4);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indices[p], p5);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indices[p], p6);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indices[p], p7);
}


void
BaseConstraint::VarChoices( const Scheme scheme,
			    const Index i1,
			    const Index i2,
			    const Index i3,
			    const Index i4,
			    const Index i5,
			    const Index i6,
			    const Index i7 )
{
   ASSERT((Index)scheme<nschemes);
   Index p=0;
   
   if (p == varCount) return;
   var_choices(scheme, p++) = i1;
   if (p == varCount) return;
   var_choices(scheme, p++) = i2;
   if (p == varCount) return;
   var_choices(scheme, p++) = i3;
   if (p == varCount) return;
   var_choices(scheme, p++) = i4;
   if (p == varCount) return;
   var_choices(scheme, p++) = i5;
   if (p == varCount) return;
   var_choices(scheme, p++) = i6;
   if (p == varCount) return;
   var_choices(scheme, p++) = i7;
}


/***************************************************************************
 * The Satisfy method is where the constraint is maintained.
 * The BaseConstraint ChooseChange method is used to select which variable
 *      should be altered to maintain the constraint.
 * Reference variables are frequently used to speed up accesses of the
 *      constraint's variables and to make the Satisfy method more legible.
 * Satisfy should return 1 if it is able to satisfy the constraint, and
 *      0 otherwise.
 */
bool
BaseConstraint::Satisfy( const Index, const Scheme, const double,
			 BaseVariable*&, VarCore& )
{
   ASSERTFAIL("BaseConstraint: Can't satisfy!");
   // return false;
}


void
BaseConstraint::print( ostream& os )
{
   unsigned int i;

   for (Index j=0; j < nschemes; j++) {
      os << name << " (" << SchemeString((Scheme)j) << ") (";
      for (i = 0; i < varCount; i++) {
	 if (i != whichMethod) {
	    os << "\t";
	    vars[i]->printc(os, var_indices[i]);
	    os << " (->" << var_choices(j, i) << ")";
	    os << endl;
	 }
      }
      os << "\t-> ";
      if (whichMethod < varCount) {
	 vars[whichMethod]->printc(os, var_indices[whichMethod]);
	 os << " (->" << var_choices(j, whichMethod) << ")";
      } else {
	 os << "(Special option.";
      }
      os << ")" << endl;
   }
}


void
BaseConstraint::printc( ostream& os, const Scheme scheme )
{
   unsigned int i;

   os << name << " (" << SchemeString(scheme) << ") (";
   os << "Called by " << callingMethod << ") (" << endl;
   for (i = 0; i < varCount; i++) {
      if (i != whichMethod) {
	 os << "\t";
	 vars[i]->printc(os, var_indices[i]);
	 os << " (->" << var_choices(scheme, i) << ")";
	 os << endl;
      }
   }
   os << "\t-> ";
   vars[whichMethod]->printc(os, var_indices[whichMethod]);
   os << " (->" << var_choices(scheme, whichMethod) << ")";
   os << ")" << endl;
}

void
BaseConstraint::Register()
{
   Index index;

   for (index = 0; index < varCount; index++)
      var_indices[index] = vars[index]->Register(this, index);
}

} // End namespace SCIRun
