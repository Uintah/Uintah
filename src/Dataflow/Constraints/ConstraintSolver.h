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

#include <Dataflow/share/share.h>
#include <Dataflow/Constraints/BaseConstraint.h>

#include <stack>
using std::stack;

namespace SCIRun {


enum RecurseType { UnInit, RecurseInitial, RecurseNormal, RecurseMax };

typedef unsigned char uchar;
struct PSECORESHARE StackItem {
   inline StackItem() : var(0), rtype(UnInit), iter(0) {}
   inline StackItem( BaseVariable* v ) : var(v), rtype(RecurseInitial), iter(0) {}
   inline StackItem( BaseVariable* v, const uchar rt, const uchar i ) : var(v), rtype(rt), iter(i) {}
   inline StackItem( const StackItem& i ) : var(i.var), rtype(i.rtype), iter(i.iter) {}
   inline ~StackItem() {}
   
   StackItem& operator=( const StackItem& i ) { var=i.var; rtype=i.rtype; iter=i.iter; return *this; }
   int operator==( const StackItem& i ) { return (var==i.var)&&(rtype==i.rtype)&&(iter==i.iter); }

   void print( std::ostream& os );
   
   BaseVariable* var;
   uchar rtype;
   uchar iter;
};

class PSECORESHARE ConstraintSolver {
public:
   ConstraintSolver();
   ConstraintSolver( const double epsilon );
   ~ConstraintSolver();
   
   void SetEpsilon( const double epsilon );
   double GetEpsilon() const;
   
   void SetMaxDepth( const Index max );
   Index GetMaxDepth() const;
   
   bool VariablesChanged() const { return changed; }
   void ResetChanged() { changed = false;}
   void SetChanged() {changed = true;}
   
   friend class BaseVariable;
private:
   void AddVariable( BaseVariable* v );
   void RemoveVariable( BaseVariable* v );
   void Solve( BaseVariable* var, const VarCore& newValue, const Scheme scheme );
   
   double Epsilon;
   Index MaxDepth;
   bool changed;
   
   vector<BaseVariable*> variables;
};

} // End namespace SCIRun


#endif
