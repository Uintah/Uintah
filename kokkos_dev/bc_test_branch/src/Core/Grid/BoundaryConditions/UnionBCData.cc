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

#include <Core/Grid/BoundaryConditions/UnionBCData.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Variables/UnionIterator.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <algorithm>

using namespace std;
using namespace SCIRun;
using namespace Uintah;

UnionBCData::UnionBCData( std::vector<BCGeomBase*> & children, const string & name, const Patch::FaceType & side ) :
  BCGeomBase( name, side )
{
  if( children.size() <= 1 ) {
    throw ProblemSetupException( "UnionBCData(): Size of children must be 2 or larger.", __FILE__, __LINE__ );
  }

  d_bcs = children[0]->d_bcs;

  for( unsigned int pos = 0; pos < children.size(); ++pos ) {
    d_children.push_back( children[ pos ] );
  }  
}

UnionBCData::UnionBCData( const UnionBCData & mybc) :
  BCGeomBase( mybc )
{
  throw ProblemSetupException( "UnionBCData(): Don't call copy constructor...", __FILE__, __LINE__ );
}

UnionBCData::~UnionBCData()
{
  // FIXME possible memory leak... 
}

bool
UnionBCData::operator==(const BCGeomBase& rhs) const
{
  const UnionBCData* p_rhs = 
    dynamic_cast<const UnionBCData*>(&rhs);

  if (p_rhs == NULL) {
    return false;
  }
  else {
    if (this->d_children.size() != p_rhs->d_children.size()) {
      return false;
    }
    return equal( this->d_children.begin(), this->d_children.end(), p_rhs->d_children.begin() );
  }
}

bool
UnionBCData::inside( const Point & p ) const 
{
  for( vector<BCGeomBase*>::const_iterator i = d_children.begin(); i != d_children.end(); ++i ) {
    if ((*i)->inside(p))
      return true;
  }
  return false;
}

void
UnionBCData::print( int depth ) const
{
  string indentation( depth*2, ' ' );

  cout << indentation << "UnionBCData: " << d_name << "\n";

  for( map<int,BCData*>::const_iterator itr = d_bcs.begin(); itr != d_bcs.end(); itr++ ) {
    itr->second->print( depth + 2 );
  }

  for( vector<BCGeomBase*>::const_iterator i = d_children.begin(); i != d_children.end(); ++i ) {
    (*i)->print( depth + 2 );
  }
}

void
UnionBCData::determineIteratorLimits( const Patch::FaceType   face, 
                                      const Patch           * patch, 
                                      const vector<Point>   & test_pts )
{
  cout << "UnionBCData::determineIteratorLimits() " << d_name << "\n"; // (this: " << this << ")\n";

  map<int,const Patch*>::const_iterator iter = d_iteratorLimitsDetermined.find( patch->getID() );

  if( iter != d_iteratorLimitsDetermined.end() ) {

    if( iter->second == patch ) {
      cout << "----warning: determineIteratorLimits called twice-----------------------------------------\n";
      cout << "patch: " << patch << "\n";
      cout << "face: "  << face << "\n";
      cout << "---------------------------------------------\n";
      return;
      throw ProblemSetupException( "DifferenceBCData()::determineIteratorLimits() called twice on patch/face...", __FILE__, __LINE__ );
    }
  }

  for( vector<BCGeomBase*>::const_iterator bc = d_children.begin(); bc != d_children.end(); ++bc ) {
    (*bc)->determineIteratorLimits( face, patch, test_pts );
  }
  
  UnionIterator cells,nodes;

  for( vector<BCGeomBase*>::const_iterator bc = d_children.begin(); bc != d_children.end(); ++bc ) {

    const Iterator & cell_itr = (*bc)->getCellFaceIterator( patch );
    const Iterator & node_itr = (*bc)->getNodeFaceIterator( patch );

    Iterator base_ci(cells),base_ni(nodes);

    cells = UnionIterator(base_ci,cell_itr);
    nodes = UnionIterator(base_ni,node_itr);
  }

  d_cells[ patch->getID() ] = scinew Iterator( UnionIterator(cells) );
  d_nodes[ patch->getID() ] = scinew Iterator( UnionIterator(nodes) );

  d_iteratorLimitsDetermined[ patch->getID() ] = patch;

  cout << "Union Limits:\n";
  printLimits();
}

// Returns a list of all the materials that the BCGeom corresponds to
set<int>
UnionBCData::getMaterials() const
{
  set<int> the_union;

  for( unsigned int pos = 0; pos < d_children.size(); pos++ ) {

    const set<int> & child_materials = d_children[pos]->getMaterials();

    the_union.insert( child_materials.begin(), child_materials.end() );
  }

  cout << "UnionBCData::getMaterials(): size of union is: " << the_union.size();

  return the_union;
}
