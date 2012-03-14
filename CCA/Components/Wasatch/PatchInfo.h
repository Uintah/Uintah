/*
 * Copyright (c) 2012 The University of Utah
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

#ifndef Wasatch_PatchInfo_h
#define Wasatch_PatchInfo_h

#include <map>

/**
 *  \file PatchInfo.h
 */

namespace SpatialOps{ class OperatorDatabase; }

namespace Wasatch{

  /**
   *  \ingroup WasatchCore
   *  \struct PatchInfo
   *  \author James C. Sutherland
   *  \date   June, 2010
   *
   *  \brief Holds information about a patch.  This is useful for
   *  individual nodes in a graph so that they have access to
   *  operators, etc.
   */
  struct PatchInfo
  {
    SpatialOps::OperatorDatabase* operators;
    int patchID;
  };

  /**
   *  \ingroup WasatchCore
   *
   *  Defines a map between the patch index (Uintah assigns this) and
   *  the PatchInfo object associated with the patch.  This is
   *  generally only required by Wasatch when pairing operators with
   *  their associated patch.
   */
  typedef std::map< int, PatchInfo > PatchInfoMap;

}

#endif // Wasatch_PatchInfo_h
