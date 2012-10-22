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

c common inlcude file with ramping function
      factor = 1.0d0
      if (ramping) then
        if (time .lt. 2.0d0) then
           factor = time*0.5d0
           if (time.lt.0.02d0) then
              factor = 0.000001d0*factor
           elseif (time.lt.0.1d0) then
              factor = 0.001d0*factor
           elseif (time.lt.0.2d0) then
              factor = 0.01d0*factor
           elseif (time.lt.0.3d0) then
              factor = 0.1d0*factor
           elseif (time.lt.0.4d0) then
              factor = 0.5d0*factor
           elseif (time.lt.1.0d0) then
              factor = 0.8d0*factor
           endif
        endif
      endif
