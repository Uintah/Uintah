/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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


#include <Core/GeometryPiece/GeometryPiece.h>

using namespace Uintah;

DebugStream GeometryPiece::gp_dbg( "GeometryPiece", "GeometryPiece", "Geometry piece debug stream", false );

//______________________________________________________________________
//
GeometryPiece::GeometryPiece() :
  m_isNameSet( false ),
  m_isFirstOutput( true )
{
}

//______________________________________________________________________
//
GeometryPiece::~GeometryPiece()
{
}

//______________________________________________________________________
//
void
GeometryPiece::outputProblemSpec( ProblemSpecP & ps ) const
{
  ProblemSpecP child_ps = ps->appendChild( getType().c_str() );

  if( m_isNameSet ) {
    child_ps->setAttribute( "label", name_ );

    if( m_isFirstOutput ) {
      // If geom obj is named, then only output data the first time.
      gp_dbg << "GP::outputProblemSpec(): Full description of: " << name_ << " -- " << getType() << "\n";
      outputHelper( child_ps );
      m_isFirstOutput = false;

    } else {
      gp_dbg << "GP::outputProblemSpec(): Reference to: " << name_ << " -- " << getType() << "\n";
    }

  } else {
    gp_dbg << "GP::outputProblemSpec(): Full Description Of: " << name_ << " -- " << getType() << "\n";
    // If no name, then always print out all data.
    outputHelper( child_ps );
  }
}
