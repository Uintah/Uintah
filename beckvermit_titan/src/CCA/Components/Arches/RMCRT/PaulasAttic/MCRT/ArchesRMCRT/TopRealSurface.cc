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

#include <iostream>
#include <CCA/Components/Arches/MCRT/ArchesRMCRT/TopRealSurface.h>

using std::cout;
using std::endl;

// i, j, k is settled at the center of the VOLUME cell
// so the corresponding control volume right below this top surface is:
// iIndex, jIndex, kIndex
// and the surface's index is:
// surfaceiIndex=iIndex,
// surfacejIndex =jIndex,
// surfacekIndex = (kIndex+1)


// inline
TopRealSurface::TopRealSurface(const int &iIndex,
			       const int &jIndex,
			       const int &kIndex,
			       const int &Ncx){
  
  surfaceiIndex = iIndex;
  surfacejIndex = jIndex;
  surfacekIndex = kIndex+1;
  // TopBottomNo = NoSurface;
  surfaceIndex = iIndex + jIndex * Ncx;
  
}


TopRealSurface::TopRealSurface(){
}


TopRealSurface::~TopRealSurface(){
}
