/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#ifndef TarAndSootInfo_h
#define TarAndSootInfo_h

namespace WasatchCore{

  /**
   *  \ingroup WasatchFields
   *  \ingroup WasatchCore
   *
   *  \class  TarAndSootInfo
   *  \author Josh McConnell
   *  \date   November, 2016
   *
   */
  class  TarAndSootInfo
  {
  public:

    static const TarAndSootInfo& self();

    // Tar properties. It is assumed that coal tar is Naphthalene.
    const double tarHydrogen,
                 tarCarbon,
                 tarMW,
                 tarDiffusivity,
                 tarHeatOfOxidation;

    // soot properties
    const double sootDensity,
                 cMin,
                 sootHeatOfOxidation;

  private:
    TarAndSootInfo();
  };
}
#endif /* TarAndSootInfo_h */
