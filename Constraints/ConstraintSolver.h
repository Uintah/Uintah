
/*
 *  ConstraintSolver.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Constraint_Solver_h
#define SCI_project_Constraint_Solver_h 1

#include <Constraints/BaseConstraint.h>
#include <Classlib/Stack.h>


enum RecurseType { UnInit, RecurseInitial, RecurseNormal, RecurseMax };

typedef unsigned char uchar;
struct StackItem {
   inline StackItem() : var(NULL), rtype(UnInit), iter(0) {}
   inline StackItem( BaseVariable* v ) : var(v), rtype(RecurseInitial), iter(0) {}
   inline StackItem( BaseVariable* v, const uchar rt, const uchar i ) : var(v), rtype(rt), iter(i) {}
   inline StackItem( const StackItem& i ) : var(i.var), rtype(i.rtype), iter(i.iter) {}
   inline ~StackItem() {}
   
   StackItem& operator=( const StackItem& i ) { var=i.var; rtype=i.rtype; iter=i.iter; return *this; }
   int operator==( const StackItem& i ) { return (var==i.var)&&(rtype==i.rtype)&&(iter==i.iter); }

   void print( ostream& os=cout ) { os<<"StackItem:  "<<var->GetName()<<","<<rtype<<","<<iter<<endl; }
   
   BaseVariable* var;
   uchar rtype;
   uchar iter;
};
ostream& operator<<( ostream& os, StackItem& i );


class ConstraintSolver {
public:
   ConstraintSolver();
   ConstraintSolver( const Real epsilon );
   ~ConstraintSolver();

   void SetEpsilon( const Real epsilon );
   Real GetEpsilon() const;

   friend class BaseVariable;
private:
   void AddVariable( BaseVariable* v );
   void RemoveVariable( BaseVariable* v );
   void Solve( BaseVariable* var, const VarCore& newValue, const Scheme scheme );

   Real Epsilon;

   Stack<StackItem> stack;

   Index NumVariables;
   Array1<BaseVariable*> variables;
};


#endif
