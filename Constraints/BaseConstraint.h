
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

#include <Constraints/manifest.h>
#include <Geometry/Point.h>
#include <Classlib/Array1.h>
#include <Classlib/Array2.h>
#include <Classlib/String.h>


/* Priority levels */
// P_constant is for a variable in reference to a constraint.
enum VPriority { P_Constant, P_Lowest, P_LowMedium,
		 P_Default, P_HighMedium, P_Highest };

enum Scheme { Scheme1, Scheme2, Scheme3, Scheme4,
	      Scheme5, Scheme6, Scheme7, Scheme8, DefaultScheme };

enum VarType { PointVar, RealVar };


class BaseVariable;

// USE THESE TO BE CLEAR!
typedef BaseVariable PointVariable;
typedef BaseVariable RealVariable;

class BaseConstraint;

class BaseVariable {
   friend class BaseConstraint;
public:
   BaseVariable( const clString& name, const Scheme s, const Point& initialValue );
   BaseVariable( const clString& name, const Scheme s, const Real initialValue );
   ~BaseVariable();

   void Order(); // Use to let the Variable order its constraints.
   void Resolve(); // Use to initially fulfill constraints.

   inline void Reset();

   // Widgets use these instead of Assign!
   void Set( const Point& newValue, const Scheme s = DefaultScheme );
   void Set( const Real newValue, const Scheme s = DefaultScheme );
   void SetDelta( const Vector& deltaValue, const Scheme s = DefaultScheme );
   void SetDelta( const Real deltaValue, const Scheme s = DefaultScheme );

   // Widgets use these to move whole widget.
   // i.e. they don't change constraints!!
   inline void Move( const Point& newValue );
   inline void Move( const Real newValue );
   inline void MoveDelta( const Vector& deltaValue );
   inline void MoveDelta( const Real deltaValue );
   
   inline const Point& GetPoint() const;
   inline Real GetReal() const;

   inline void SetEpsilon( const Real epsilon );
   inline Real GetEpsilon() const;
   
   void print( ostream& os=cout );

   // Constraints use this instead of Set!
   // ONLY Constraints use this...
   void Assign( const Point& newValue, const Scheme scheme );
   void Assign( const Real newValue, const Scheme scheme );

private:
   clString name;
   VarType vartype;
   
   Point pointvalue;
   Real realvalue;
   
   Real Epsilon;

   int levellevel, level;

   Index numconstraints;
   Array1<BaseConstraint*> constraints;
   Array1<Index> constraint_indexs;
   Array1<VPriority> constraint_priorities;
   Array1<Index> constraint_order;
   
   Scheme scheme;

   inline Index Register( BaseConstraint* constraint, const Index index );
   inline void RegisterPriority( const Index index, const VPriority p );
   void printc( ostream& os, const Index c );
};
inline ostream& operator<<( ostream& os, BaseVariable& v );


class BaseConstraint {
   friend class BaseVariable;
public:
   BaseConstraint( const clString& name, const Index numSchemes,
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

   void print( ostream& os=cout );
   void printc( ostream& os, const Scheme scheme );

protected:
   clString name;
   Index nschemes;
   
   Index varCount;
   Array1<BaseVariable*> vars;
   Array1<Index> var_indexs; // The var's index for this constraint.
   Array2<Index> var_choices;
   Index whichMethod, callingMethod;

   void Register();
   virtual inline Index ChooseChange( const Index index, const Scheme scheme );
   virtual void Satisfy( const Index index, const Scheme scheme );
};
inline ostream& operator<<( ostream& os, BaseVariable& v );


/* Miscellaneous */
char* PriorityString( const VPriority p );
char* SchemeString( const Scheme s );

inline int ConstantVariable( const VPriority p )
{
   return (p == P_Constant);
}


inline int HigherPriority( const VPriority p1, const VPriority p2 )
{
   return (p1 > p2);
}


inline void
BaseVariable::Reset()
{
   levellevel = level = 0;
}


inline void
BaseVariable::Move( const Point& newValue )
{
   ASSERT(vartype==PointVar);
   pointvalue = newValue;
}


inline void
BaseVariable::Move( const Real newValue )
{
   ASSERT(vartype==RealVar);
   realvalue = newValue;
}


inline void
BaseVariable::MoveDelta( const Vector& deltaValue )
{
   ASSERT(vartype==PointVar);
   pointvalue += deltaValue;
}


inline void
BaseVariable::MoveDelta( const Real deltaValue )
{
   ASSERT(vartype==RealVar);
   realvalue += deltaValue;
}


inline void
BaseVariable::SetEpsilon( const Real epsilon )
{
   Epsilon = epsilon;
}


inline Real
BaseVariable::GetEpsilon() const
{
   return Epsilon;
}


inline Index
BaseVariable::Register( BaseConstraint* constraint, const Index index )
{
   constraints.grow(1);
   constraints[numconstraints] = constraint;
   constraint_indexs.grow(1);
   constraint_indexs[numconstraints] = index;
   constraint_priorities.grow(1);
   constraint_priorities[numconstraints] = P_Default;
   constraint_order.grow(1);
   constraint_order[numconstraints] = 0;
   numconstraints++;

   return numconstraints-1;
}


inline void
BaseVariable::RegisterPriority( const Index index, const VPriority p )
{
   ASSERT(index < numconstraints);

   if (p == P_Constant)
      Error("Variable given constant priority for constraint.");

   constraint_priorities[index] = p;
}


inline const Point&
BaseVariable::GetPoint() const
{
   ASSERT(vartype==PointVar);
   return pointvalue;
}


inline Real
BaseVariable::GetReal() const
{
   ASSERT(vartype==RealVar);
   return realvalue;
}


inline Index
BaseConstraint::ChooseChange( const Index index, const Scheme scheme )
{
   return whichMethod = var_choices(scheme, callingMethod = index);
}


inline ostream&
operator<<( ostream& os, BaseVariable& v )
{
   v.print(os);
   return os;
}


inline ostream&
operator<<( ostream& os, BaseConstraint& c )
{
   c.print(os);
   return os;
}


#endif
