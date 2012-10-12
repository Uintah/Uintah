/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <cmath>

const int ISOTROPIC = 0; // isotropic scattering, phase function = 1
const int LINEAR_SCATTER = 1;
const int EDDINGTON = 2;

const int TOP = 0;
const int BOTTOM = 1;
const int FRONT = 2;
const int BACK = 3;
const int LEFT = 4;
const int RIGHT = 5;
const double pi=  4 * atan(1);
const double SB = 5.669 * pow(10., -8);

// all normal vectors pointing inward the volume
extern const double n_top[3];
extern const double n_bottom[3];
extern const double n_front[3];
extern const double n_back[3];
extern const double n_left[3];
extern const double n_right[3];
extern const double *surface_n[6];

  


