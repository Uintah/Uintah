/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <Core/Grid/Ghost.h>

using namespace Uintah;


std::string Ghost::names[numGhostTypes] ={
  "None",
  "nodes",
  "cells",
  "x faces",
  "y faces",
  "z faces",
  "all faces",
  "xpypzp",
  "xpypzm",
  "xpymzp",
  "xpymzm",
  "xmypzp",
  "xmypzm",
  "xmymzp",
  "xmymzm",
};


IntVector Ghost::directions[numGhostTypes] =
{
  IntVector(0,0,0),     // None
  IntVector(1,1,1),     // AroundNodes
  IntVector(1,1,1),     // AroundCells
  IntVector(1,0,0),     // AroundFacesX
  IntVector(0,1,0),     // AroundFacesY
  IntVector(0,0,1),     // AroundFacesZ
  IntVector(1,1,1),     // AroundFaces
  // One sided ghosts
  IntVector(1,1,1),     // xpypzp
  IntVector(1,1,0),     // xpypzm
  IntVector(1,0,1),     // xpymzp
  IntVector(1,0,0),     // xpymzm
  IntVector(0,1,1),     // xmypzp
  IntVector(0,1,0),     // xmypzm
  IntVector(0,0,1),     // xmymzp
  IntVector(0,0,0)      // xmymzm
};




