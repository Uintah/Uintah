
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

#include <PSECore/share/share.h>
#include <PSECore/Constraints/BaseConstraint.h>
#include <SCICore/Containers/Stack.h>

namespace PSECore {
namespace Constraints {

using SCICore::Containers::Stack;

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

   void print( ostream& os );
   
   BaseVariable* var;
   uchar rtype;
   uchar iter;
};

class PSECORESHARE ConstraintSolver {
public:
   ConstraintSolver();
   ConstraintSolver( const Real epsilon );
   ~ConstraintSolver();
   
   void SetEpsilon( const Real epsilon );
   Real GetEpsilon() const;
   
   void SetMaxDepth( const Index max );
   Index GetMaxDepth() const;
   
   int VariablesChanged() const {return changed;}
   void ResetChanged() {changed = 0;}
   void SetChanged() {changed = 1;}
   
   friend class BaseVariable;
private:
   void AddVariable( BaseVariable* v );
   void RemoveVariable( BaseVariable* v );
   void Solve( BaseVariable* var, const VarCore& newValue, const Scheme scheme );
   
   Real Epsilon;
   Index MaxDepth;
   int changed;
   
   Stack<StackItem> stack;
   
   Index NumVariables;
   Array1<BaseVariable*> variables;
};

} // End namespace Constraints
} // End namespace PSECore

//
// $Log$
// Revision 1.4  1999/09/08 02:26:38  sparker
// Various #include cleanups
//
// Revision 1.3  1999/08/26 23:57:02  moulding
// changed SCICORESHARE to PSECORESHARE
//
// Revision 1.2  1999/08/17 06:38:17  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:54  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:06  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//

#endif
