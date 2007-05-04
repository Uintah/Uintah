
#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace std;

static DebugStream dbg( "GeometryPiece", false );

GeometryPiece::GeometryPiece() :
  nameSet_( false ),
  firstOutput_( true )
{
}

GeometryPiece::~GeometryPiece()
{
}

void
GeometryPiece::outputProblemSpec( ProblemSpecP & ps ) const
{
  ProblemSpecP child_ps = ps->appendChild( getType().c_str() );

  if( nameSet_ ) {
    child_ps->setAttribute( "label", name_ );

    if( firstOutput_ ) {
      // If geom obj is named, then only output data the first time.
      dbg << "GP::outputProblemSpec(): Full description of: " << name_ << " -- " << getType() << "\n";
      outputHelper( child_ps );
      firstOutput_ = false;

    } else {
      dbg << "GP::outputProblemSpec(): Reference to: " << name_ << " -- " << getType() << "\n";
    }

  } else {
    dbg << "GP::outputProblemSpec(): Full Description Of: " << name_ << " -- " << getType() << "\n";
    // If no name, then always print out all data.
    outputHelper( child_ps );
  }
}

