#ifndef WRAP_NRRD_H
#define WRAP_NRRD_H

#include <teem/nrrd.h>

#include <Packages/Uintah/StandAlone/tools/uda2nrrd/Matrix_Op.h>

template<class FIELD> Nrrd * wrap_nrrd( FIELD * source, Matrix_Op matrix_op, bool verbose );

#endif // WRAP_NRRD_H
