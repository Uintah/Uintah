/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <Core/Grid/BoundaryConditions/DifferenceBCData.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Variables/DifferenceIterator.h>
#include <Core/Malloc/Allocator.h>

#include <set>
#include <iostream>
#include <algorithm>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

DifferenceBCData::DifferenceBCData( BCGeomBase * p1, BCGeomBase * p2, const string & name, const Patch::FaceType & side ) :
  BCGeomBase( name, side ),
  d_left(  p1 ),
  d_right( p2 )
{
  d_bcs = p1->d_bcs;

    cout << "DifferenceBCData(p1, p2, name, side) called for " << name << "\n";
    cout << "    p1: " << p1->getName() << /*", " << p1 <<*/ ", " << p1->getSide() << "\n";
    cout << "    p2: " << p2->getName() << /*", " << p2 <<*/ ", " << p2->getSide() << "\n";
    //    cout << "this: " << this << "\n";
  
    print(5);
}

DifferenceBCData::~DifferenceBCData()
{
  // FIXME: possible memory leak? (for d_bcs)
}

DifferenceBCData& DifferenceBCData::operator=(const DifferenceBCData& rhs)
{
  throw ProblemSetupException( "DifferenceBCData(): Error, don't call operator=.", __FILE__, __LINE__ );
  return *this;
}

bool
DifferenceBCData::operator==(const BCGeomBase& rhs) const
{
  const DifferenceBCData* p_rhs = 
    dynamic_cast<const DifferenceBCData*>(&rhs);

  if (p_rhs == NULL)
    return false;
  else
    return (this->d_left == p_rhs->d_left) && (this->d_right == p_rhs->d_right);
}

bool
DifferenceBCData::inside( const Point & p ) const 
{
  return( d_left->inside(p) && !d_right->inside(p) );
}

void
DifferenceBCData::print( int depth ) const
{
  string indentation( depth*2, ' ' );
  string indent2( depth*2+2, ' ' );

  cout << indentation << "DifferenceBCData: " << d_name << "\n"; //, " << this << "\n";

  for( map<int,BCData*>::const_iterator itr = d_bcs.begin(); itr != d_bcs.end(); itr++ ) {
    itr->second->print( depth + 1 );
  }

  cout << indent2 << "Left:\n";

  d_left->print( depth + 2 );

  cout << indent2 << "Right:\n";

  d_right->print( depth + 2 );
}

void
DifferenceBCData::determineIteratorLimits( const Patch::FaceType   face,
                                           const Patch           * patch,
                                           const vector<Point>   & test_pts )
{
  cout << "DifferenceBCData::determineIteratorLimits(): " << d_name << "\n"; // " (this: " << this << ")\n";

  map<int,const Patch*>::const_iterator iter = d_iteratorLimitsDetermined.find( patch->getID() );

  if( iter != d_iteratorLimitsDetermined.end() ) {

    if( iter->second == patch ) {
      cout << "---warning: determineIteratorLimits called twice on this patch ------------------------------------------\n";
      cout << "patch: " << patch->getID() << "\n";
      cout << "face: "  << face << "\n";
      cout << "---------------------------------------------\n";
      return;
      throw ProblemSetupException( "DifferenceBCData()::determineIteratorLimits() called twice on patch/face...", __FILE__, __LINE__ );
    }
  }

  cout << "DifferenceBCData::determineIteratorLimits(): " << d_name << "\n"; // (this: " << this << ")\n";

  d_left->determineIteratorLimits(  face, patch, test_pts );
  d_right->determineIteratorLimits( face, patch, test_pts );

  d_left->determineIteratorLimits(  face, patch, test_pts ); // FIXME: when should this really be called?
  d_right->determineIteratorLimits( face, patch, test_pts ); // FIXME: when should this really be called?

  const Iterator & left_cell  = d_left->getCellFaceIterator( patch );
  const Iterator & left_node  = d_left->getNodeFaceIterator( patch );
  const Iterator & right_cell = d_right->getCellFaceIterator( patch );
  const Iterator & right_node = d_right->getNodeFaceIterator( patch );

  d_cells[ patch->getID() ] = scinew Iterator( DifferenceIterator( left_cell, right_cell ) );
  d_nodes[ patch->getID() ] = scinew Iterator( DifferenceIterator( left_node, right_node ) );

  d_iteratorLimitsDetermined[ patch->getID() ] = patch;

  cout << "End Difference Limits:\n"; // " << this << "\n";
  printLimits();
}

// Returns a list of all the materials that the BCGeom corresponds to
set<int>
DifferenceBCData::getMaterials() const
{
  set<int> left_materials, right_materials, the_union;

  left_materials  = d_left->getMaterials();
  right_materials = d_right->getMaterials();

  the_union.insert( left_materials.begin(), left_materials.end() );
  the_union.insert( right_materials.begin(), right_materials.end() );

  cout << "DifferenceBCData::getMaterials(): size of union is: " << the_union.size();

  return the_union;
}
