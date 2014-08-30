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

#ifndef SpatialOps_Configure_h
#define SpatialOps_Configure_h

/*
 * Configuration options for SpatialOps.
 */

/* #undef ENABLE_THREADS */
#ifdef ENABLE_THREADS
# define NTHREADS 1
#else
# define NTHREADS 1
#endif

/* #undef ENABLE_CUDA */
/* #undef NEBO_REPORT_BACKEND */
/* #undef NEBO_GPU_TEST */

/* #undef CUDA_ARCHITECTURE */

#define SOPS_REPO_DATE "Tue Aug 26 14:26:26 2014 -0600"
#define SOPS_REPO_HASH "d2708b1de00853da4d48077742ad263518631fbd"

#endif // SpatialOps_Configure_h
