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


#ifndef SCI_project_Base_Constraint_h
#define SCI_project_Base_Constraint_h 1

#include <Dataflow/Constraints/BaseVariable.h>
#include <Core/Containers/Array2.h>
#include <string>
#include <vector>

namespace SCIRun {
using std::string;
using std::vector;


class BaseConstraint {
   friend class BaseVariable;
   friend class ConstraintSolver;
public:
   BaseConstraint( const string& name, const Index numSchemes,
		   const Index VariableCount );
   virtual ~BaseConstraint();

   // Use this to define the priorities of this constraint in relation
   // to each of its variables.
   // (This is a cheating way for varargs...)
   void Priorities( const VPriority p1 = P_Default,
		    const VPriority p2 = P_Default,
		    const VPriority p3 = P_Default,
		    const VPriority p4 = P_Default,
		    const VPriority p5 = P_Default,
		    const VPriority p6 = P_Default,
		    const VPriority p7 = P_Default );
   // Use this to define the variable to change to fulfill the constraint
   // given the variable that requested re-satisfication.
   // (This is a cheating way for varargs...)
   void VarChoices( const Scheme scheme,
		    const Index i1 = 0,
		    const Index i2 = 0,
		    const Index i3 = 0,
		    const Index i4 = 0,
		    const Index i5 = 0,
		    const Index i6 = 0,
		    const Index i7 = 0 );

   void print( std::ostream& os );
   void printc( std::ostream& os, const Scheme scheme );

protected:
   string name;
   Index nschemes;
   
   Index varCount;
   vector<BaseVariable*> vars;
   vector<Index> var_indices; // The var's index for this constraint.
   Array2<Index> var_choices;
   Index whichMethod, callingMethod;

   void Register();
   inline Index ChooseChange( const Index index, const Scheme scheme );
   virtual bool Satisfy( const Index index, const Scheme scheme,
			 const double Epsilon,
			 BaseVariable*& var, VarCore& c );
};
inline std::ostream& operator<<( std::ostream& os, BaseConstraint& v );


/***************************************************************************
 * The BaseConstraint ChooseChange method is used to select which variable
 *      should be altered to maintain the constraint.
 */
inline Index
BaseConstraint::ChooseChange( const Index index, const Scheme scheme )
{
   return whichMethod = var_choices(scheme, callingMethod = index);
}

inline std::ostream&
operator<<( std::ostream& os, BaseConstraint& c )
{
   c.print(os);
   return os;
}

} // End namespace SCIRun


#endif
