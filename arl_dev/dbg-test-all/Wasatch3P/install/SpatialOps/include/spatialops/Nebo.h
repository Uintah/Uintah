/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#ifndef NEBO_H
#define NEBO_H

#include <spatialops/SpatialOpsConfigure.h>

#include <spatialops/NeboBasic.h>
#include <spatialops/NeboRhs.h>
#include <spatialops/NeboMask.h>
#include <spatialops/NeboOperators.h>
#include <spatialops/NeboCond.h>
#include <spatialops/NeboStencils.h>
#include <spatialops/NeboLhs.h>
#include <spatialops/NeboStencilBuilder.h>
#include <spatialops/NeboAssignment.h>
#include <spatialops/NeboReductions.h>


/* NEBO_ERROR_TRAP can be used to find where Nebo errors originate.
 * NEBO_ERROR_TRAP takes a string and a block of code to run.
 * NEBO_ERROR_TRAP does nothing in Release mode.
 * In Debug mode, NEBO_ERROR_TRAP does the following:
 *  If there is no error in the given code, NEBO_ERROR_TRAP does nothing.
 *  If there is an error in the given code, NEBO_ERROR_TRAP adds the current file and line number.
 *
 * For example:
 *  NEBO_ERROR_TRAP("TEST LOCATION", throw(std::runtime_error("ERROR MESSAGE"));)
 * will throw this error:
 *  terminate called after throwing an instance of 'std::runtime_error'
 *    what():  Error in NEBO ERROR TRAP (ERROR LOCATION)
            ERROR MESSAGE
 *          At: /uufs/chpc.utah.edu/common/home/u0623470/aurora-cwearl/code/SpatialOps/test/Nebo.h : 48
 *  Aborted
 */
#ifndef NDEBUG
#define NEBO_ERROR_TRAP(STRING, CODE)     \
  try { CODE }                            \
  catch(std::runtime_error& e){           \
    std::ostringstream msg;               \
    msg << "Error in NEBO ERROR TRAP (";  \
    msg << STRING;                        \
    msg << ")\n\t" << e.what();           \
    msg << "\n\tAt: ";                    \
    msg << __FILE__ << " : " << __LINE__; \
    throw(std::runtime_error(msg.str())); \
  };
#else
#define NEBO_ERROR_TRAP(STRING, CODE) CODE
#endif

#endif

