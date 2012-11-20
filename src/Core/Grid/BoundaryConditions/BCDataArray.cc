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

#include <Core/Grid/BoundaryConditions/BoundCond.h> // FIXME, used for testing, delete

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Grid/BoundaryConditions/DifferenceBCData.h>
#include <Core/Grid/BoundaryConditions/SideBCData.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <algorithm>
#include <set>
#include <vector>
#include <functional>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

// export SCI_DEBUG="BCDA_DBG:+"
static DebugStream BCDA_dbg( "BCDA_DBG", false );

////////////////////////////////////////////////////////////////////////////////////

BCDataArray::BCDataArray() 
{
}

BCDataArray::BCDataArray( const BCDataArray & rhs )
{
  bcDataArrayType::const_iterator mat_id_itr;
  for( mat_id_itr = rhs.d_BCDataArray.begin(); mat_id_itr != rhs.d_BCDataArray.end(); ++mat_id_itr ) {
    int mat_id = mat_id_itr->first;
    const vector<BCGeomBase*>& rhs_vec = mat_id_itr->second;
    vector<BCGeomBase*>& bcDataArray_vec = d_BCDataArray[mat_id];
    vector<BCGeomBase*>::const_iterator vec_itr;
    for( vec_itr = rhs_vec.begin(); vec_itr != rhs_vec.end(); ++vec_itr ) {

      BCGeomBase * bcb = (*vec_itr);

      bcDataArray_vec.push_back( bcb );
    }
  }
}

// FIXME Dd: Not sure if I want to do this copying...
#if 0
BCDataArray *
BCDataArray::copy( const BCDataArray & bcda )
{
  bcDataArrayType::const_iterator mat_id_itr;
  for( mat_id_itr = bcda.d_BCDataArray.begin(); mat_id_itr != bcda.d_BCDataArray.end(); ++mat_id_itr ) {
    int mat_id = mat_id_itr->first;
    const vector<BCGeomBase*>& bcda_vec = mat_id_itr->second;
    vector<BCGeomBase*>& bcDataArray_vec = d_BCDataArray[mat_id];
    vector<BCGeomBase*>::const_iterator vec_itr;
    for( vec_itr = bcda_vec.begin(); vec_itr != rhs_vec.end(); ++vec_itr ) {

      BCGeomBase * bcb = (*vec_itr);

      bcDataArray_vec.push_back( bcb );
    }
  }
}
#endif

BCDataArray::~BCDataArray()
{
  // cout << "Calling BCDataArray destructor" << endl;

  // FIXME - possible memory leak?
}

////////////////////////////////////////////////////////////////////////////////////

BCDataArray&
BCDataArray::operator=(const BCDataArray& rhs)
{
  throw ProblemSetupException( "Don't use operator= for BCDataArray....", __FILE__, __LINE__ );
  return *this;
}

void
BCDataArray::determineIteratorLimits(       Patch::FaceType   face,
                                      const Patch           * patch )
{
  IntVector lpts,hpts;
  patch->getFaceCells(face,-1,lpts,hpts);
  vector<Point> test_pts;

  cout << "BCDataArray::determineIteratorLimits() for " << this << ", f/p: " << face << ", " << patch << "\n";
  print();

  for (CellIterator candidatePoints(lpts,hpts); !candidatePoints.done(); candidatePoints++) {
    IntVector nodes[8];
    patch->findNodesFromCell(*candidatePoints,nodes);
    Point pts[8];
    Vector p( 0.0, 0.0, 0.0 );
    for (int i = 0; i < 8; i++) {
      pts[i] = patch->getLevel()->getNodePosition(nodes[i]);
    }

    if      (face == Patch::xminus) {
      p = (pts[0].asVector()+pts[1].asVector()+pts[2].asVector() + pts[3].asVector())/4.;
    }
    else if (face == Patch::xplus) {
      p = (pts[4].asVector()+pts[5].asVector()+pts[6].asVector() + pts[7].asVector())/4.;
    }
    else if (face == Patch::yminus) {
      p = (pts[0].asVector()+pts[1].asVector()+pts[4].asVector() + pts[5].asVector())/4.;
    }
    else if (face == Patch::yplus) {
      p = (pts[2].asVector()+pts[3].asVector()+pts[6].asVector() + pts[7].asVector())/4.;
    }
    else if (face == Patch::zminus) {
      p = (pts[0].asVector()+pts[2].asVector()+pts[4].asVector() + pts[6].asVector())/4.;
    }
    else if (face == Patch::zplus) {
      p = (pts[1].asVector()+pts[3].asVector()+pts[5].asVector() + pts[7].asVector())/4.;
    }
    else {
      throw ProblemSetupException( "Invalid face.", __FILE__, __LINE__ );
    }

    test_pts.push_back(Point(p.x(),p.y(),p.z()));
  }
  
  BCDataArray::bcDataArrayType::iterator mat_id_itr;
  for( mat_id_itr = d_BCDataArray.begin(); mat_id_itr != d_BCDataArray.end(); ++mat_id_itr ) {
    vector<BCGeomBase*>& bc_objects = mat_id_itr->second;

    cout << "here: " << mat_id_itr->first << "\n";

    for( vector<BCGeomBase*>::iterator iter = bc_objects.begin(); iter != bc_objects.end(); ++iter ) {
      BCGeomBase * bcgb = *iter;
      bcgb->determineIteratorLimits( face, patch, test_pts );
    }
  }
} // end determineIteratorLimits()

void
BCDataArray::addBCGeomBase( BCGeomBase * bc )
{
  set<int> materials = bc->getMaterials();

  cout << "BCDataArray::addBCGeomBase:  before adding bc\n";
  print();

  cout << "Adding this BC:\n";
  bc->print();

  cout << "going to add these materials:\n";
  for( set<int>::const_iterator iter = materials.begin(); iter != materials.end(); iter++ ) {
    cout << *iter << " ";
  }
  cout << "\n";

  for( set<int>::const_iterator iter = materials.begin(); iter != materials.end(); iter++ ) {
    d_BCDataArray[ *iter ].push_back( bc );
  }

  cout << "BCDataArray::addBCGeomBase:  AFTER adding bc\n";
  print();

}

#if 0
void
BCDataArray::combineBCGeometryTypes_NEW(int mat_id)
{
  if (d_BCDataArray[mat_id].size() <= 1) {
    cout << "One or fewer elements in BCDataArray\n\n";
    return;
  }

  vector<BCGeomBase*>& bcDataArray_vec = d_BCDataArray[mat_id];
  
  vector<BCGeomBase*> new_bcdata_array;
  // Look to see if there are duplicate SideBCData types, if so, then
  // combine them into one (i.e. copy the BCData from the duplicate into
  // the one that will actually be stored).

  //  count the number of unique geometry types

  vector<BCGeomBase*>::iterator v_itr,nv_itr;

  for( v_itr = bcDataArray_vec.begin(); v_itr != bcDataArray_vec.end(); ++v_itr ) {
    cout << "number of SideBCData = " << 
      count_if( bcDataArray_vec.begin(), bcDataArray_vec.end(), cmp_type<SideBCData>() ) << "\n";
  }

  if( count_if(bcDataArray_vec.begin(), bcDataArray_vec.end(), cmp_type<SideBCData>()) > 1 ) {
    cout << "Have duplicates Before...\n";
    for( v_itr = bcDataArray_vec.begin(); v_itr != bcDataArray_vec.end(); ++v_itr ) {
      cout << "type of element = " << typeid(*(*v_itr)).name() << endl;
      (*v_itr)->print();
    }
    cout << "\n\n";
  }

  // Put the last element in the bcDataArray_vec into the new_bcdata_array
  // and delete this element

  BCGeomBase* element = bcDataArray_vec.back();
  BCGeomBase* clone_element = element->clone();
  
  new_bcdata_array.push_back(clone_element);
  delete element;
  bcDataArray_vec.pop_back();

  while( !bcDataArray_vec.empty() ) {
    element = bcDataArray_vec.back();
    bool foundit = false;
    for( nv_itr = new_bcdata_array.begin(); nv_itr != new_bcdata_array.end(); ++nv_itr ) {
      if( *(*nv_itr) == *element ) {
        foundit = true;
        break;
      }
    }
    if( foundit ) {
      bcDataArray_vec.pop_back();
      const BCData & bcd = element->getBC();
      const BCData & n_bcd = (*nv_itr)->getBC();
      // n_bcd.combine(bcd);  FIXME

      //      (*nv_itr)->addBC(n_bcd);    FIXME

      delete element;
    } else {
      new_bcdata_array.push_back(element->clone());
      bcDataArray_vec.pop_back();
      delete element;
    }

  }

  cout << "Size of new_bcdata_array = " << new_bcdata_array.size() << "\n";
  cout << "Size of bcDataArray_vec = " << bcDataArray_vec.size() << "\n";

  for( nv_itr = new_bcdata_array.begin(); nv_itr != new_bcdata_array.end(); ++nv_itr ) {
    (*nv_itr)->print();
    cout << "\n\n";
  }

  for_each( bcDataArray_vec.begin(), bcDataArray_vec.end(), delete_object<BCGeomBase>() );

  bcDataArray_vec.clear();
#if 1
  bcDataArray_vec = new_bcdata_array;
#endif

} // end combineBCGeometryTypes_NEW()
#endif

const BoundCondBase* 
BCDataArray::getBoundCondData( int mat_id, const string & type, int ichild ) const
{
  const BoundCondBase * result = NULL;

  BCGeomBase   * child = getChild( mat_id, ichild );
  const BCData * bcd   = child->getBCData( mat_id );

  if( bcd == NULL ) {
    bcd = child->getBCData( -1 );
  }

  result = bcd->getBCValue( type );

  if( result == NULL ) { // see if it is a generic  FIXME is this right? qwerty qwerty qwerty 
    bcd = child->getBCData( -1 );    
    result = bcd->getBCValue( type );
  }

  if( result == NULL ) {
    const BCData * bcd = child->getBCData( -1 ); // all materials
    result = bcd->getBCValue( "Auxiliary" );
  }
  return result;


#if 0
  bcDataArrayType::const_iterator itr = d_BCDataArray.find( mat_id );

  if( itr == d_BCDataArray.end() ) {
    // Given material does not have any explicit BCs, so check -1...

    const vector<BCGeomBase*> & bcgbv = itr->second;

    const BCData * bcd = bcgbv[ichild]->getBCData( -1 ); // all materials

    result = bcd->getBCValue( type );
  }
  else {

    const vector<BCGeomBase*> & bcgbv = itr->second;  //d_BCDataArray[ mat_id ];

    const BCData * bcd = bcgbv[ ichild ]->getBCData( mat_id );

    if( bcd->exists( type ) ) {
      return bcd->getBCValue( type );
    }
    else {
      bcd = bcgbv[ ichild ]->getBCData( mat_id );
      result = bcd->getBCValue( type );
    }
  }

  if( result == NULL ) {
    const BCData * bcd = itr->second[ichild]->getBCData( -1 ); // all materials
    result = bcd->getBCValue( "Auxiliary" );
  }

  // FIXME debugging
  if( result == NULL ) {
    throw ProblemSetupException( "result is null...", __FILE__, __LINE__ );
  }
  // end FIXME debugging

  return result;
#endif
}

const
Iterator &
BCDataArray::getCellFaceIterator( int mat_id, int ichild, const Patch * patch ) const
{
  BCGeomBase * geom = getChild( mat_id, ichild );
  return geom->getCellFaceIterator( patch );
#if 0
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[ichild]->getCellFaceIterator(b_ptr);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end()) {
      const BCGeomBase * bcgb = itr->second[ichild];
      bcgb->print();

      bcgb->getCellFaceIterator( b_ptr );
    }
  }
#endif
}

const
Iterator &
BCDataArray::getNodeFaceIterator( int mat_id, int ichild, const Patch * patch ) const
{
  BCGeomBase * geom = getChild( mat_id, ichild );
  return geom->getNodeFaceIterator( patch );
#if 0
  bcDataArrayType::const_iterator itr = d_BCDataArray.find(mat_id);
  if (itr != d_BCDataArray.end()) {
    itr->second[ichild]->getNodeFaceIterator(b_ptr);
  }
  else {
    itr = d_BCDataArray.find(-1);
    if (itr != d_BCDataArray.end())
      itr->second[ichild]->getNodeFaceIterator(b_ptr);
  }
#endif
}

int
BCDataArray::getNumberChildren( int mat_id ) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find( mat_id );

  if( mat_id != -1 && itr != d_BCDataArray.end() ) {

    int num_in_specific_matl = itr->second.size();

    itr = d_BCDataArray.find( -1 );
    int num_in_generic_matl  = itr->second.size();

    return num_in_generic_matl + num_in_specific_matl;
  }
  else {
    itr = d_BCDataArray.find( -1 );
    return itr->second.size();
  }
}

BCGeomBase*
BCDataArray::getChild( int mat_id, int child_index ) const
{
  bcDataArrayType::const_iterator itr = d_BCDataArray.find( mat_id );

  if( mat_id == -1 || itr == d_BCDataArray.end() ) { // If mat_id -1 is explicitly asked for, or if the explicit mat_id doesn't exist...

    if( mat_id != -1 ) {
      itr = d_BCDataArray.find( -1 );
    }

    return itr->second[ child_index ];
  }
  else { 

    // Find the BCGeomBase* in the specified material, but if the 'child_index' is outside of the 
    // specific materials array, then we need to go into the generic ('all'/-1 material) materials vector.

    int the_size = itr->second.size();
    if( child_index < the_size ) {
      return itr->second[ child_index ];
    }
    else {
      itr = d_BCDataArray.find( -1 );
      return itr->second[ child_index - the_size ];
    }
  }
}

void
BCDataArray::print() const
{
  bcDataArrayType::const_iterator bcda_itr;

  cout << "BCDataArray (size: " << d_BCDataArray.size() << "):  [" << this << "]\n";

  for( bcda_itr = d_BCDataArray.begin(); bcda_itr != d_BCDataArray.end(); bcda_itr++ ) {
    cout << "mat_id = " << bcda_itr->first << "\n";
    cout << "Size of BCGeomBase vector = " << bcda_itr->second.size() << "\n";

    const vector<BCGeomBase*> & bcgbs = bcda_itr->second;

    for( unsigned int pos = 0; pos < bcgbs.size(); ++pos ) {
      bcgbs[ pos ]->print();
    }
  }
}
