#ifndef UINTAH_DEFS_H
#define UINTAH_DEFS_H

/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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






#define NO_ARCHES
#define NO_ICE
#define NO_MPM


#define NO_WASATCH


#define FORTRAN_UNDERSCORE_END

#define RAY_SCATTER 1

#define SCIRUN_SRCDIR "/home/alan/uintah/uintah-dev/src"
#define SCIRUN_OBJDIR "/home/alan/uintah/uintah-dev/opt-gpu"

#if !defined( FIX_NAME )
#  if defined( FORTRAN_UNDERSCORE_END )
     // This ## magic (apparently) concatenates the _ to the 'fun' varaible.
#    define FIX_NAME(fun) fun ## _
#  else // NONE
#    define FIX_NAME(fun) fun
#  endif
#endif

#endif // UINTAH_DEFS_H
