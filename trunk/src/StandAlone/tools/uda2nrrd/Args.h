#ifndef UDA2NRRD_ARGS_H
#define UDA2NRRD_ARGS_H

#include <StandAlone/tools/uda2nrrd/Matrix_Op.h>

struct Args {
  bool      use_all_levels;
  bool      verbose;
  bool      quiet;
  bool      attached_header;
  bool      remove_boundary;
  Matrix_Op matrix_op;

  Args() {
    use_all_levels = false;
    verbose = false;
    quiet = false;
    attached_header = true;
    remove_boundary = false;
    matrix_op = None;
  }

};

#endif // UDA2NRRD_ARGS_H


