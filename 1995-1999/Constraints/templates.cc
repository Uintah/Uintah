
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/Array2.cc>
#include <Classlib/Stack.cc>
#include <Constraints/ConstraintSolver.h>

template class Stack<StackItem>;

class BaseConstraint;
template class Array1<BaseConstraint*>;
class BaseVariable;
template class Array1<BaseVariable*>;
template class Array1<StackItem>;
enum VPriority;
template class Array1<VPriority>;

template class Array2<unsigned int>;
