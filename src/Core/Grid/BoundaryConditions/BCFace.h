/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_GRID_BCFACE_H
#define UINTAH_GRID_BCFACE_H

#include <vector>
#include <string>

#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Grid/Patch.h>

namespace Uintah {

/*!

\class BCFace

\brief Holds information for the 6 (x-, x+, y-, etc) faces.
  
\author J. Davison de St. Germain \n
        University of Utah \n
*/

class BCFace {
public:

  BCFace( const Patch::FaceType & side );
  ~BCFace();

  void addGeometry( BCGeomBase * geom );

  const std::string              getName() const { return Patch::getFaceName( d_faceSide ); }
  const Patch::FaceType          getSide() const { return d_faceSide; }

  const std::vector<BCGeomBase*> getGeoms() const { return d_geoms; }

  // Should be called only once, after all the BCGeomBases have been added.
  // If there are more then one BCGeomBase, then a Difference between the 'side'
  // and the Union of all other BCs will replace the raw d_geoms.
  void                  combineBCs();

  void                  print();

private:
  std::vector<BCGeomBase*> d_geoms;
  Patch::FaceType          d_faceSide;

  bool                     d_finalized; // True when combineBCs has been called.

  /// Copy constructor - Made private as it should not be used.
  BCFace( const BCFace& rhs ) {}

  BCFace& operator=( const BCFace& rhs ) { return *this; } // Made private as '=' operator should not be used.

};

} // End namespace Uintah

#endif
