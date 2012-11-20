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

#include <Core/Grid/BoundaryConditions/BCFace.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/SideBCData.h>
#include <Core/Grid/BoundaryConditions/UnionBCData.h>
#include <Core/Grid/BoundaryConditions/DifferenceBCData.h>

#include <iostream>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

BCFace::BCFace( const Patch::FaceType & side ) :
  d_faceSide( side ), d_finalized( false )
{
}

BCFace::~BCFace()
{
}

void
BCFace::addGeometry( BCGeomBase * geom )
{
  if( d_finalized ) {
    throw ProblemSetupException( "BCFace::combineBCs: Cannot addGeometry() as BCFace is already finalized...", __FILE__, __LINE__ );
  }

  ///// DEBUG FIXME
  //cout << "BCFace: adding: \n";
  //geom->print(0);
  ///// end DEBUG FIXME

  d_geoms.push_back( geom );
}

void
BCFace::combineBCs()
{
  d_finalized = true;
  std::vector<BCGeomBase*> new_geoms;

  if( d_geoms.size() > 1 ) {
    // Create a Difference Object that contains:
    //     The 'side' itself, and
    //     A Union of everything else.

    BCGeomBase * left = d_geoms[ 0 ];

    if( dynamic_cast<SideBCData*>( left ) == NULL ) {
      throw ProblemSetupException( "BCFace::combineBCs: 1st BCGeomBase must be a SideBCData...", __FILE__, __LINE__ );
    }

    BCGeomBase * right;
    string name;

    if( d_geoms.size() > 2 ) {
      vector<BCGeomBase*> children;
      for( unsigned int pos = 1; pos < d_geoms.size(); pos++ ) {
        children.push_back( d_geoms[ pos ] );
      }

      name = "BCFace auto-created Union";
      right = scinew UnionBCData( children, name, d_faceSide );

    }
    else {
      right = d_geoms[ 1 ];
    }

    name = "BCFace auto-created Difference";
    DifferenceBCData * difference = scinew DifferenceBCData( left, right, name, d_faceSide );

    //delete( d_geoms[0] );    // Clean up memory for the original 'side', then...
    d_geoms[0] = difference; // ...replace it with the new 'difference' object.
  }

  // FIXME debug statements:
  // cout << "BCFace after combineBCs():\n";
  // print();
}

void
BCFace::print()
{
  cout << "BCFace (" << (d_finalized ? "Finalized" : "Not Finalized" ) 
       << ") for side " << d_faceSide << " has the following BCGeomBases:\n";
  for( unsigned int pos = 0; pos < d_geoms.size(); pos++ ) {
    d_geoms[ pos ]->print();
  }
}
