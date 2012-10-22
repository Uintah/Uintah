/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UDA2NRRD_ARGS_H
#define UDA2NRRD_ARGS_H

#include <StandAlone/tools/uda2nrrd/Matrix_Op.h>

struct Args {
  bool      use_all_levels;
  bool      verbose;
  bool      quiet;
  bool      attached_header;
  bool      remove_boundary;
  bool      force_overwrite;
  Matrix_Op matrix_op;

  Args() {
    use_all_levels = false;
    verbose = false;
    quiet = false;
    attached_header = true;
    remove_boundary = false;
    force_overwrite = false;
    matrix_op = None;
  }

};

#endif // UDA2NRRD_ARGS_H


