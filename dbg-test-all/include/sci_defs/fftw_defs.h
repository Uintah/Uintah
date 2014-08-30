/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef SCI_FFTW_DEFS_H
#define SCI_FFTW_DEFS_H

#define HAVE_FFTW 1

#ifdef HAVE_FFTW
// When <complex.h> or <complex> is included before <fftw3.h>, then fftw_complex is
// defined to be the native complex type

// From: http://www.fftw.org/doc/Complex-numbers.html
//  C++ has its own complex<T> template class, defined in the standard <complex> header file. 
//  Reportedly, the C++ standards committee has recently agreed to mandate that the storage
//  format used for this type be binary-compatible with the C99 type, i.e. an array T[2] with
//  consecutive real [0] and imaginary [1] parts. If you have a variable std::complex<double> *x,
//  you can pass it directly to FFTW via reinterpret_cast<fftw_complex*>(x).

#  include <complex>

#  include <fftw3.h>
#  include <fftw3-mpi.h>
#endif

#endif

