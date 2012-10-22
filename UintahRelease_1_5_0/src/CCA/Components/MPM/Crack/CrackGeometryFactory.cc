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

#include <CCA/Components/MPM/Crack/CrackGeometryFactory.h>
#include <CCA/Components/MPM/Crack/CrackGeometry.h>
#include <CCA/Components/MPM/Crack/NullCrack.h>
#include <CCA/Components/MPM/Crack/QuadCrack.h>
#include <CCA/Components/MPM/Crack/CurvedQuadCrack.h>
#include <CCA/Components/MPM/Crack/TriangularCrack.h>
#include <CCA/Components/MPM/Crack/ArcCrack.h>
#include <CCA/Components/MPM/Crack/EllipticCrack.h>
#include <CCA/Components/MPM/Crack/PartialEllipticCrack.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
using std::cerr;

using namespace Uintah;

CrackGeometry* CrackGeometryFactory::create(ProblemSpecP& ps)
{
  ProblemSpecP child = ps->findBlock("crack");
  if(!child)
    return scinew NullCrack(ps);

  for (ProblemSpecP crack_segment_ps = child->findBlock(); 
       crack_segment_ps != 0; 
       crack_segment_ps = crack_segment_ps->findNextBlock()) {
    string crack_type = crack_segment_ps->getNodeName();

    if (crack_type == "quad")
      return scinew QuadCrack(crack_segment_ps);

    else if (crack_type == "curved_quad")
      return scinew CurvedQuadCrack(crack_segment_ps);

    else if (crack_type == "triangle")
      return scinew TriangularCrack(crack_segment_ps);

    else if (crack_type == "arc")
      return scinew ArcCrack(crack_segment_ps);

    else if (crack_type == "ellipse")
      return scinew EllipticCrack(crack_segment_ps);

    else if (crack_type == "partial_ellipse")
      return scinew PartialEllipticCrack(crack_segment_ps);

    else 
      throw ProblemSetupException("Unknown Crack Segment Type R ("+crack_type+")", __FILE__, __LINE__);
  }

  return 0;
}


