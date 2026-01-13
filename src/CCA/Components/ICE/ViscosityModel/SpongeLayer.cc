/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#include <CCA/Components/ICE/ViscosityModel/SpongeLayer.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/DOUT.hpp>

#include <iostream>
#include <sstream>
#include <cstdio>

using namespace Uintah;
using namespace std;

//______________________________________________________________________
//
SpongeLayer::SpongeLayer( ProblemSpecP & sLayer_ps,
                          const GridP  & grid )
 : Viscosity(sLayer_ps, grid)
{

  setCallOrder( Last );

  ProblemSpecP box_ps = sLayer_ps->findBlock( "box" );
  if( box_ps == nullptr ){
    throw ProblemSetupException("ERROR: ICE:  Couldn't find <SpongeLayer> -> <box> tag", __FILE__, __LINE__);
  }

  map<string,string> attribute;
  box_ps->getAttributes(attribute);
  m_SL_name = attribute["label"];

  setName( "SpongeLayer " + m_SL_name );

  Point min;
  Point max;
  box_ps->require( "min", min );
  box_ps->require( "max", max );
  sLayer_ps->require( "maxDynamicViscosity", m_maxDynViscosity );

  if( m_maxDynViscosity > 0.0 ){
    set_isViscosityDefined( true );
  }

  //__________________________________
  //      bulletproofing
  double near_zero = 10 * DBL_MIN;
  double xdiff =  std::fabs( max.x() - min.x() );
  double ydiff =  std::fabs( max.y() - min.y() );
  double zdiff =  std::fabs( max.z() - min.z() );

  if ( xdiff < near_zero   ||
       ydiff < near_zero   ||
       zdiff < near_zero ) {
    std::ostringstream warn;
    warn << "\nERROR: SpongeLayer: box ("<< m_SL_name
         << ") The max coordinate cannot be <= the min coordinate (max " << max << " <= min " << min << ")." ;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //__________________________________
  //  Box cannot exceed comutational boundaries
  BBox b;
  grid->getInteriorSpatialRange( b );

  const Point bmin = b.min();
  const Point bmax = b.max();

  proc0cout << "______________________________________________________________________\n"
            << " SpongeLayer (" << m_SL_name << "):\n";

  if( min.x() < bmin.x() || min.y() < bmin.y() || min.x() < bmin.z() ||
      max.x() > bmax.x() || max.y() > bmax.y() || max.x() > bmax.z() ){

    proc0cout << " WARNING:  The extents of the box exceed the computational domain.\n"
              << "           Resizing the box so it doesn't exceed the domain.\n";

  }

  min = Max( b.min(), min );
  max = Min( b.max(), max );
  m_box = Box(min,max);
  proc0cout << "  Box Dimensions: " << m_box << "\n"
            << "______________________________________________________________________\n";
}

//______________________________________________________________________
//
SpongeLayer::~SpongeLayer(){};

//______________________________________________________________________
//
void SpongeLayer::outputProblemSpec(ProblemSpecP& vModels_ps)
{
  ProblemSpecP vModel = vModels_ps->appendChild("Model");
  vModel->setAttribute("name", "SpongeLayer");

  vModel->appendElement("maxDynamicViscosity", m_maxDynViscosity);
  ProblemSpecP box_ps = vModel->appendChild("box");
  box_ps->setAttribute( "label", m_SL_name );
  box_ps->appendElement( "min", m_box.lower() );
  box_ps->appendElement( "max", m_box.upper() );

}
//______________________________________________________________________
//        Determine the cell indicies
void SpongeLayer::initialize( const Level * level)
{
  m_lowIndx  = findCell( level, m_box.lower() );
  m_highIndx = findCell( level, m_box.upper() );

  // bulletproofing
  IntVector diff = m_highIndx - m_lowIndx;

  if ( diff.x() < 1 || diff.y() < 1 || diff.z() < 1 ){
    std::ostringstream warn;
    warn << "\nERROR: Sponge Layer: box ("<< m_SL_name
           << ") There must be at least one computational cell difference between the max and min ("<< diff << ")";
   throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//       Return the cell iterator over all cells in the sponge layer on this patch
CellIterator SpongeLayer::getCellIterator( const Patch* patch ) const {

  IntVector lo = Max(m_lowIndx,  patch->getCellLowIndex());
  IntVector hi = Min(m_highIndx, patch->getCellHighIndex());

  return CellIterator( lo, hi );
}

//______________________________________________________________________
//        Find the cell index closest to the point
IntVector SpongeLayer::findCell( const Level * level,
                                 const Point & p)
{
  Vector  dx     = level->dCell();
  Point   anchor = level->getAnchor();
  Vector  v( (p - anchor) / dx);

  IntVector cell (roundNearest(v));  // This is slightly different from Level implementation
  return cell;
}

//______________________________________________________________________
//        Set the viscosity for each cell in the region
//        If the viscosity will be non-zero set the flag.
template< class CCVar>
void SpongeLayer::computeDynViscosity_impl( const Patch       * patch ,
                                            CCVar              & ,
                                            CCVariable<double> & mu)
{
  CellIterator iter = getCellIterator( patch );

  //  set the flag if the viscosity is > 0 somewhere on this patch.
  bool flag = false;

  if( iter.size() > 0  && m_maxDynViscosity > 0.0){
    flag = true;
  }
  set_isViscosityDefined(flag);


  for (;!iter.done();iter++) {
    IntVector c = *iter;
    mu[c] = m_maxDynViscosity;
  }
}


//______________________________________________________________________
//        string that shows the extents of the box
std::string
SpongeLayer::getExtents_string() const
{
  ostringstream mesg;
  mesg.setf(ios::scientific,ios::floatfield);
  mesg.precision(4);
  mesg << "  SpongeLayer (" << m_SL_name
      << ") box lower: " << m_box.lower() << " upper: " << m_box.upper()
      <<", lowIndex: "  << m_lowIndx << ", highIndex: " << m_highIndx;
  mesg.setf(ios::scientific ,ios::floatfield);

  return mesg.str();
}

//______________________________________________________________________
//
void
SpongeLayer::print()
{
  DOUTR( true, getExtents_string() );
}


