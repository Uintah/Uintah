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
 *  BaseVariable.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Base_Variable_h
#define SCI_project_Base_Variable_h 1

#include <Dataflow/share/share.h>

#include <Dataflow/Constraints/VarCore.h>
#include <string>
#include <vector>

namespace SCIRun {
using std::string;
using std::vector;

/* Priority levels */
// P_constant is for a variable in reference to a constraint.
enum VPriority { P_Lowest, P_LowMedium, P_Default, P_HighMedium, P_Highest };

enum Scheme { Scheme1, Scheme2, Scheme3, Scheme4,
	      Scheme5, Scheme6, Scheme7, Scheme8, DefaultScheme };

// USE THESE TO BE CLEAR!
#define PointVariable BaseVariable
#define RealVariable BaseVariable

class BaseConstraint;

class PSECORESHARE BaseVariable {
   friend class BaseConstraint;
   friend class ConstraintSolver;
public:
   BaseVariable( const string& name, ConstraintSolver* s, const Scheme scheme,
		 const Point& initialValue );
   BaseVariable( const string& name, ConstraintSolver* s, const Scheme scheme,
		 const double initialValue );
   ~BaseVariable();

   void Order(); // Use to let the Variable order its constraints.

   // Widgets use these instead of Assign!
   void Set( const Point& newValue, const Scheme s = DefaultScheme );
   void Set( const double newValue, const Scheme s = DefaultScheme );
   void SetDelta( const Vector& deltaValue, const Scheme s = DefaultScheme );
   void SetDelta( const double  deltaValue, const Scheme s = DefaultScheme );

   // Widgets use these to move whole widget.
   // i.e. they don't change constraints!!
   void Move( const Point& newValue );
   void Move( const double newValue );
   void MoveDelta( const Vector& deltaValue );
   void MoveDelta( const double deltaValue );
   
   inline Point point() const;
   inline operator Point() const;
   inline double real() const;
   inline operator double() const;

   void print( std::ostream& os );

   inline const string& GetName() const { return name; }
   inline int GetNumConstraints() const { return numconstraints; }

private:
   string name;

   VarCore data;

   ConstraintSolver* solver;
   
   unsigned int level;

   Index numconstraints;
   vector<BaseConstraint*> constraints;
   vector<Index>           constraint_indices;
   vector<VPriority>       constraint_priorities;
   vector<Index>           constraint_order;
   
   Scheme scheme;

   Index Register( BaseConstraint* constraint, const Index index );
   void RegisterPriority( const Index index, const VPriority p );
   void printc( std::ostream& os, const Index c );
};
inline PSECORESHARE std::ostream& operator<<( std::ostream& os, BaseVariable& v );


/* Miscellaneous */
char* PriorityString( const VPriority p );
char* SchemeString( const Scheme s );

inline PSECORESHARE int HigherPriority( const VPriority p1, const VPriority p2 )
{
   return (p1 > p2);
}


inline Point
BaseVariable::point() const
{
   return (Point)data;
}


inline
BaseVariable::operator Point() const
{
   return (Point)data;
}


inline double
BaseVariable::real() const
{
   return (double)data;
}


inline
BaseVariable::operator double() const
{
   return (double)data;
}


inline PSECORESHARE std::ostream&
operator<<( std::ostream& os, BaseVariable& v )
{
   v.print(os);
   return os;
}

} // End namespace SCIRun


#endif
