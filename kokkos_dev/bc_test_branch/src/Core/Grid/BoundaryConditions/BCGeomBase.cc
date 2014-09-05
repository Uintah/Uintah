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

#include <Core/Grid/BoundaryConditions/BCGeomBase.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/GridIterator.h>
#include <Core/Grid/Variables/ListOfCellsIterator.h>
#include <Core/Grid/Level.h>

#include <iostream>
#include <vector>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

///////////////////////////////////////////////////////////////////////////////

void
BCGeomBase::init( const string & name, const Patch::FaceType & side )
{
  //  d_cells    = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
  //  d_nodes    = GridIterator(IntVector(0,0,0),IntVector(0,0,0));
  d_name     = name;
  d_faceSide = side;
}

BCGeomBase::BCGeomBase( const string & name, const Patch::FaceType & side )
{
  init( name, side );
}

BCGeomBase::BCGeomBase()
{
  init();
}

BCGeomBase::~BCGeomBase()
{
}

// Copy constructor
BCGeomBase::BCGeomBase( const BCGeomBase & rhs )
{
  throw ProblemSetupException( "BCGeomBase() copy constructor should not be used...", __FILE__, __LINE__ );
}

///////////////////////////////////////////////////////////////////////////////

BCGeomBase& BCGeomBase::operator=(const BCGeomBase& rhs)
{
  throw ProblemSetupException( "BCGeomBase::operator= should not be used...", __FILE__, __LINE__ );
  return *this;
}

///////////////////////////////////////////////////////////////////////////////

const Iterator &
BCGeomBase::getCellFaceIterator( const Patch * patch ) const
{
  map<int,const Patch*>::const_iterator iter = d_iteratorLimitsDetermined.find( patch->getID() );

  if( iter == d_iteratorLimitsDetermined.end() ) {
    throw ProblemSetupException( "BCGeomBase()::getCellFaceIterator() called before iterators determined!", __FILE__, __LINE__ );
  }

  // FIXME DEBUG STATEMENTS
  //  cout <<"BCGeomBase::getCellFaceIterator():\n";
  //  cout << "d_cells = "; d_cells.limits( cout ); cout << "\n";

  map<int,Iterator*>::const_iterator c_iter = d_cells.find( patch->getID() );

  cout << "getCellFaceIterator\n";
  cout << "first:  " << c_iter->first << "\n";
  cout << "second: " << *(c_iter->second) << "\n";

  return *(c_iter->second);
}

const Iterator &
BCGeomBase::getNodeFaceIterator( const Patch * patch ) const
{
  map<int,const Patch*>::const_iterator iter = d_iteratorLimitsDetermined.find( patch->getID() );

  if( iter == d_iteratorLimitsDetermined.end() ) {
    throw ProblemSetupException( "BCGeomBase()::getNodeFaceIterator() called before iterators determined!", __FILE__, __LINE__ );
  }

  map<int,Iterator*>::const_iterator n_iter = d_cells.find( patch->getID() );

  return *(n_iter->second);
}

void
BCGeomBase::addBC( const BoundCondBase * bc ) 
{
  BCData * bcdata = d_bcs[ bc->getMatl() ];

  if( bcdata == NULL ) {
    bcdata = new BCData();
  }

  bcdata->addBC( bc );

  d_bcs[ bc->getMatl() ] = bcdata;
}

const BCData *
BCGeomBase::getBCData( int matl ) const
{

  map<int,BCData*>::const_iterator itr = d_bcs.find( matl );

  if( itr == d_bcs.end() ) 
    return NULL;
  else
    return itr->second;
}

set<int>
BCGeomBase::getMaterials() const
{
  set<int> materials;

  for( map<int,BCData*>::const_iterator itr = d_bcs.begin(); itr != d_bcs.end(); itr++ ) {

    materials.insert( itr->first );
  }

  return materials;
}

void
BCGeomBase::determineIteratorLimits( const Patch::FaceType   face,
                                     const Patch           * patch,
                                     const vector<Point>   & test_pts )
{
  // FIXME DEBUG STATEMENTS
  cout << "BCGeomBase::determineIteratorLimits() for " << patch->getFaceName( face )<< "\n";
  cout << "     BCGeom name: " << d_name << ", side: " << d_faceSide << "\n"; // (this: " << this << ")\n";
  cout << "     patch: "/* << patch*/ << ", patch id is: " << patch->getID() << "\n";

  if( d_iteratorLimitsDetermined[ patch->getID() ] != NULL ) {

    // NOTE: we don't (currently) have to check to see if the same face is computed more than once
    // because for each face there is a separate array of these objects, and thus they are
    // only determined once...


    cout << "FIXME: warning, determineIteratorLimits already called\n";
    return;

    //throw ProblemSetupException( "BCGeomBase()::determineIteratorLimits() already called...", __FILE__, __LINE__ );

    //   FIXME: Possible that this throw isn't necessary but using it at least for now
    //          to determine who is making this call...
    //   throw ProblemSetupException( "BCGeomBase()::determineIteratorLimits() called twice...", __FILE__, __LINE__ );
  }

  d_iteratorLimitsDetermined[ patch->getID() ] = patch;

  IntVector l,h;
  patch->getFaceCells(face,0,l,h);
  GridIterator cells(l,h);

  IntVector ln,hn;
  patch->getFaceNodes(face,0,ln,hn);
  GridIterator nodes(ln,hn);

  Iterator cell_itr(cells), node_itr(nodes);

  vector<Point>::const_iterator pts = test_pts.begin();

  ListOfCellsIterator list_cells;
  vector<IntVector> vec_cells;

  for (cell_itr.reset(); !cell_itr.done();cell_itr++,pts++) {
    if (inside(*pts)) {
      vec_cells.push_back(*cell_itr);
    }
  }
  
  ListOfCellsIterator list_nodes;
  vector<IntVector> vec_nodes;

  for (node_itr.reset(); !node_itr.done();node_itr++) {
    Point p = patch->getLevel()->getNodePosition(*node_itr);
    if (inside(p)) {
      vec_nodes.push_back(*node_itr);
    }
  }

  if (vec_cells.empty()) {
    cout << "here: id is " << patch->getID() << ", vec_cells empty " << "\n";
    d_cells[ patch->getID() ] = scinew Iterator( GridIterator(IntVector(0,0,0),IntVector(0,0,0)) );
  }
  else {
    for( vector<IntVector>::const_iterator i = vec_cells.begin(); i != vec_cells.end(); ++i ) {
      list_cells.add(*i);
    }

    cout << "here: id is " << patch->getID() << ", num of cells is " << list_cells.size() << "\n";

    d_cells[ patch->getID() ] = scinew Iterator( list_cells );
  }
  
  if( vec_nodes.empty() ) {
    d_nodes[ patch->getID() ] = scinew Iterator( GridIterator( IntVector(0,0,0),IntVector(0,0,0) ) );
  }
  else {
    for (vector<IntVector>::const_iterator i = vec_nodes.begin(); i != vec_nodes.end(); ++i) {
      list_nodes.add( *i );
    }
    d_nodes[ patch->getID() ] = scinew Iterator( list_nodes );
  }
}

void
BCGeomBase::printLimits() const
{
  cout << "FIXME: printlimits() not implemented yet\n";

  //  cout << "d_cells = "; d_cells.limits( cout ); cout << " (this: " << this << ")\n";
  //  cout << "d_nodes = "; d_nodes.limits( cout ); cout << " (detemined?: " << d_iteratorLimitsDetermined << "\n";
}
