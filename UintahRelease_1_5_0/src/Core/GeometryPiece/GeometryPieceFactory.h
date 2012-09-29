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

#ifndef __GEOMETRY_OBJECT_FACTORY_H__
#define __GEOMETRY_OBJECT_FACTORY_H__

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/GeometryPiece/GeometryPiece.h>

#include   <vector>
#include   <map>
#include   <string>

#include <Core/GeometryPiece/GeometryPiece.h>
namespace Uintah {

  class GeometryPieceFactory
  {
  public:
    // this function has a switch for all known go_types
    // and calls the proper class' readParameters()
    // addMaterial() calls this
    static void create(const ProblemSpecP& ps,
		       std::vector<GeometryPieceP>& objs);

    // Clears out the saved geometry piece information...  In theory, this should
    // only be called by (and necessary for) the Switcher component (and only if 
    // a component that is being switched to happens to be a 'copy' of a previous component).
    static void resetFactory();

    // Runs through all the GeometryPiece that have been created and
    // sets their flag for first time output.  This should be done at
    // the beginning of any output of a problemspec.
    static void resetGeometryPiecesOutput();

  private:
    // This variable records all named GeometryPieces, so that if they
    // are referenced a 2nd time, they don't have to be rebuilt.
    //
    // Assuming multiple GeometryPieceFactory's will not exist and if
    // they do, they won't be executing at the same time (in different
    // threads)... If this is not the case, then this variable should
    // be locked...
    static std::map<std::string,GeometryPieceP> namedPieces_;
    static std::vector<GeometryPieceP>          unnamedPieces_;
  };

} // End namespace Uintah

#endif /* __GEOMETRY_PIECE_FACTORY_H__ */
