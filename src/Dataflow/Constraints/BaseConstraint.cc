//static char *id="@(#) $Id$";

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
#include <PSECore/Constraints/BaseConstraint.h>

namespace PSECore {
namespace Constraints {

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 * The BaseConstraint constructor allocates the constraint's standard
 *      structures.
 */
BaseConstraint::BaseConstraint( const clString& name, const Index nschemes,
				const Index varCount )
: name(name), nschemes(nschemes), varCount(varCount),
  vars(varCount), var_indexs(varCount), var_choices(nschemes, varCount)
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
   vars[p]->RegisterPriority(var_indexs[p], p1);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p], p2);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p], p3);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p], p4);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p], p5);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p], p6);
   p++;
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p], p7);
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
   ASSERT(scheme<nschemes);
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
int
BaseConstraint::Satisfy( const Index, const Scheme, const Real, BaseVariable*&, VarCore& )
{
   ASSERTFAIL("BaseConstraint: Can't satisfy!");
   return 0;
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
	    vars[i]->printc(os, var_indexs[i]);
	    os << " (->" << var_choices(j, i) << ")";
	    os << endl;
	 }
      }
      os << "\t-> ";
      if (whichMethod < varCount) {
	 vars[whichMethod]->printc(os, var_indexs[whichMethod]);
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
	 vars[i]->printc(os, var_indexs[i]);
	 os << " (->" << var_choices(scheme, i) << ")";
	 os << endl;
      }
   }
   os << "\t-> ";
   vars[whichMethod]->printc(os, var_indexs[whichMethod]);
   os << " (->" << var_choices(scheme, whichMethod) << ")";
   os << ")" << endl;
}

void
BaseConstraint::Register()
{
   Index index;

   for (index = 0; index < varCount; index++)
      var_indexs[index] = vars[index]->Register(this, index);
}

} // End namespace Constraints
} // End namespace PSECore

//
// $Log$
// Revision 1.3  1999/08/18 20:20:17  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:38:15  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:53  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//
