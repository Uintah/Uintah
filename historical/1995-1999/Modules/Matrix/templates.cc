
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Datatypes/MatrixPort.h>

class TCLstring;
template class Array1<TCLstring*>;
class TCLint;
template class Array1<TCLint*>;
template class Array1<MatrixIPort*>;
template class Array1<MatrixHandle>;
