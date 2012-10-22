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
/*
 *    File   : sci_comp_warn_fixes.h
 *    Author : J. Davison de St. Germain
 *    Date   : Aug. 25, 2005
 *
 *    This is a list of macros that can be used to fix compiler
 *    warnings that, when fixed, unfortunately cause a different
 *    compiler warning on another platform.
 *
 */

#if defined(__sgi) && !defined(__GNUC__)

   // This macro is used in procedures that can not reach the end of the
   // function, and thus the SGI compiler warns that you have a
   // 'unreachable' statement.  However, if you don't have it under gnu,
   // then you get a 'this function must return a value' warning.
#  define RETURN_FALSE
#  define RETURN_NULL
#  define RETURN_0
#  define RETURN

   // This removes compiler warnings about unreachable statements.
   // Only use BREAK after lines that will kill the program.  Otherwise
   // use the normal 'break'.
#  define BREAK 

#else

#  define RETURN_FALSE return false
#  define RETURN_NULL return NULL
#  define RETURN_0 return 0
#  define RETURN   return

#  define BREAK break
#endif

