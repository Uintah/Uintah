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


#ifndef Packages_Uintah_CCA_Components_Examples_RegionDB_h
#define Packages_Uintah_CCA_Components_Examples_RegionDB_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/GridP.h>

#include   <map>
#include   <string>

namespace Uintah {

  class RegionDB {

  public:
    RegionDB();
    ~RegionDB() {}
    void problemSetup(ProblemSpecP& ps, const GridP& grid);
    GeometryPieceP getObject(const std::string& name) const;

  private:
    void addRegion(GeometryPieceP piece);
    void addRegion(GeometryPieceP piece, const std::string& name);

    typedef std::map<std::string, GeometryPieceP> MapType;
    MapType db;
  };
}

#endif
