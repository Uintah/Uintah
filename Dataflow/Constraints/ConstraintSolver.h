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

#include <Dataflow/Constraints/BaseConstraint.h>

#include <stack>

namespace SCIRun {
using std::stack;


typedef unsigned char uchar;
struct StackItem {
   inline StackItem() : var(0), iter(0) {}
   inline StackItem( BaseVariable* v ) : var(v), iter(0) {}
   inline StackItem( BaseVariable* v, const uchar i ) : var(v), iter(i) {}
   inline StackItem( const StackItem& i ) : var(i.var), iter(i.iter) {}
   inline ~StackItem() {}
   
   StackItem& operator=( const StackItem& i ) { var=i.var; iter=i.iter; return *this; }
   int operator==( const StackItem& i ) { return (var==i.var)&&(iter==i.iter); }

   void print( std::ostream& os );
   
   BaseVariable* var;
   uchar iter;
};

class ConstraintSolver {
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
