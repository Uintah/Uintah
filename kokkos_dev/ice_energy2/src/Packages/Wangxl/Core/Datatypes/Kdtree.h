#ifndef SCI_Wangxl_Datatypes_Kdtree_h
#define SCI_Wangxl_Datatypes_Kdtree_h

#include <list>
#include <queue>
#include <utility>
#include <algorithm>
#include <math.h>

namespace Wangxl {

using std::list;
using std::sort;
using std::priority_queue;
using std::pair;

#ifndef LARGENUMBER
#define LARGENUMBER 1e300
#endif
#ifndef SMALLNUMBER
#define SMALLNUMBER -1e300
#endif

template < class TPoint, class TPointElem >
class KDTree {
  template< class NODETYPE >
  class TreeNode {
  public:
    TreeNode(const NODETYPE &d )
  : leftPtr( 0 ), data( d ), rightPtr( 0 ) {}
    NODETYPE getData() const { return data; }
    
    TreeNode < NODETYPE > *leftPtr; // pointer to left subtree
    NODETYPE data;
    TreeNode < NODETYPE > *rightPtr; //pointer to right subtree
  };

  class Data {
  public:
    Data( TPoint* value ) { point = value; }
    Data( TPointElem value ) { split = value; }
    inline TPoint* getPoint() { return point; }
    inline TPointElem getSplit() { return split; }
  private:
    TPoint* point;
    TPointElem split;
  };

  class compare {
    int _dim;
    int _dimensions;
  public:
    compare( int dim, int dimensions ) : _dim( dim ), _dimensions( dimensions ){}
    bool operator() ( TPoint *p0, TPoint *p1 )
      { 
	return comparePoints( p0, p1 ) < 0 ;
      }
  private:
    int comparePoints( TPoint *p0, TPoint *p1 )
      {
	int dim;
	if ( (*p0)[_dim] < (*p1)[_dim] ) return -1;
	else if ( (*p0)[_dim] > (*p1)[_dim] ) return 1;
	else { //return (int)( p0 - p1 );
	  dim = ( _dim + 1 ) % _dimensions;
	  if ( (*p0)[dim] < (*p1)[dim] ) return -1;
	  else if ( (*p0)[dim] > (*p1)[dim] ) return 1;
	  else {
	    dim =  ( _dim + 2 ) % _dimensions;
	    if ( (*p0)[dim] < (*p1)[dim] ) return -1;
	    else if ( (*p0)[dim] > (*p1)[dim] ) return 1;
	    else { /* overlaping points*/ return 0; }
	  }
	}
      }
  };
  
  TreeNode < Data* > *rootPtr;
  int dimensions;
public:
  KDTree();
  KDTree( list< TPoint* >& points, int dim ) { 
    TPoint **v;
    list< TPoint* >::const_iterator i;
    int index = 0;
    int n = (int) points.size();
    v = new TPoint*[n];
    for ( i = points.begin(); i != points.end(); i++ ) v[index++] = *i;
    dimensions = dim;
    rootPtr = buildKDTree( v, n, 0 );
    delete v;
  }

  // Range Query
  void query( const TPoint& low, const TPoint& high, list< TPoint* >& found ) {
    TPoint regionLow, regionHigh;
    TPointElem radius = 0;
    int i;
    
    for( i = 0; i < dimensions; i++ ) {
      regionLow.set( i, SMALLNUMBER );
      regionHigh.set( i, LARGENUMBER );
    }
    searchKDTree( rootPtr, regionLow, regionHigh, 0, low, high, radius, 0, found );
  }

  // Sphere Range Query
  void querySphere( const TPoint& point, const TPointElem radius, list< TPoint* >& found ) {
    TPoint regionLow, regionHigh, low, high;
    int i;
    
    low = high = point;
    for( i = 0; i < dimensions; i++ ) {
      regionLow.set( i, SMALLNUMBER );
      regionHigh.set( i, LARGENUMBER );
    }
    searchKDTree( rootPtr, regionLow, regionHigh, 1, low, high, radius*radius, 0, found );
  }

  // k neareast neighbor search
  void queryNearest( const TPoint& qpoint, const int k, list< TPoint* >& found ) {
    int i;
    for( i = 0; i < dimensions; i++ ) {
      lowB.set( i, SMALLNUMBER );
      highB.set( i, LARGENUMBER );
    }
    stop = false;
    searchKDTree( rootPtr, qpoint, k, 0 );
    while( !knn.empty() ) {
      found.push_back( knn.top().first );
      knn.pop();
    }
  }
private:
  TreeNode< Data* >* buildKDTree( TPoint *points[], int number, int depth ){
    int dimension;
    int half;
    int number0, number1;
    TPointElem split;
    TPoint **points0, **points1;
    TreeNode< Data* > *leftNode, *rightNode, *node;
    Data* data;
    
    if ( number == 1 ) { // leaf node, storing the point's pointer
      data = new Data( *points );
      node = new TreeNode< Data* >( data );
      return node;
    }
    half = (int)(number / 2.0 + 0.5 );
    dimension = depth % dimensions;
    sort( points, points + number, compare( dimension, dimensions ) );
    split = (*points[half-1])[dimension];
    points0 = points;
    number0 = half;
    points1 = points + half;
    number1 = number - number0;  
    leftNode = buildKDTree( points0, number0, depth + 1 );
    rightNode = buildKDTree( points1, number1, depth + 1 );
    data = new Data( split );
    node = new TreeNode< Data* >( data );
    node->leftPtr = leftNode;
    node->rightPtr = rightNode;
    return node;
  }

  // recursive search for Range Query
  void searchKDTree( TreeNode< Data* >* ptr, TPoint& regionLow, TPoint& regionHigh, const int range, const TPoint& low, const TPoint& high, TPointElem squareradius, int depth, list< TPoint* >& found ) {
    TPoint *point;
    TPointElem split;
    TPoint regionlcLow, regionlcHigh, regionrcLow, regionrcHigh;
    int inter;
    
    if ( ptr->leftPtr == 0  &&  ptr->rightPtr == 0 ) { // leaf node
      point = ptr->data->getPoint();
      inter = intersect( range, *point, *point, low, high, squareradius );
      if( inter == 1 || inter == 2 ) found.push_back( point ); // get it if this point is in or on the range
    }
    else { // nonleaf nodes
      regionlcLow = regionrcLow = regionLow;
      regionlcHigh = regionrcHigh = regionHigh;
      split = ptr->data->getSplit();
      if ( ptr->leftPtr ) { // there is a left node
	regionlcHigh.set( depth % dimensions, split );
	inter = intersect( range, regionlcLow, regionlcHigh, low, high, squareradius );
	if ( inter == 2 ) add( ptr->leftPtr, found ); 
	else if ( inter == 1 || inter == 3 ) // this region intersect or contain the range
	  searchKDTree( ptr->leftPtr, regionlcLow, regionlcHigh, range, low, high, squareradius, depth + 1, found );
      }
      if ( ptr->rightPtr) { // there is a right node
	regionrcLow.set( depth % dimensions, split );
	inter =  intersect( range, regionrcLow, regionrcHigh, low, high, squareradius );
	if ( inter == 2 ) add( ptr->rightPtr, found );
	else if ( inter == 1 || inter == 3 ) // this region intersect or contain the range
	  searchKDTree( ptr->rightPtr, regionrcLow, regionrcHigh, range, low, high,  squareradius, depth + 1, found );
      }
    }
  }

  // Box-Sphere testiong 0: seperate; 1: intersect; 2: box in sphere; 3: sphere in box. range = 0: Box vs. Box; range = 1: Box vs. Sphere
  int intersect( const int range, const TPoint& regionLow, const TPoint& regionHigh,  const TPoint& low,  const TPoint& high, const TPointElem squareradius ) const {
    int i;
    bool flag = false;
    if ( range == 0 ) { // rectangle query
      for ( i = 0; i < dimensions; i++ ) {
	if( regionLow[i] > high[i] || regionHigh[i] < low[i] ) return 0;// outside
	else if ( regionLow[i] < low[i] || regionHigh[i] > high[i] ) flag = true;// intersect possible
      }
      if( flag ) return 1; // intersect
      else return 2; // inside
    }
    else  { // ( range == 1 ) sphere query
      TPointElem a, b, dmin, dmax;
      dmin = 0;
      dmax = 0;
      for( i = 0; i < dimensions; i++ ) {
	if ( fabs( regionLow[i] ) >= LARGENUMBER ) a = LARGENUMBER;
	else a = pow( ( low[i] - regionLow[i] ), 2.0 );
	if ( fabs( regionHigh[i] ) >= LARGENUMBER ) b = LARGENUMBER;
	else b = pow( ( low[i] - regionHigh[i] ), 2.0 );
	dmax += max( a, b );
	if( low[i] < regionLow[i] ) { // here low[i] is C[i]
	  if ( a > squareradius ) return 0; // outside each other
	  dmin += a;
	  flag = true;
	}
	else if( low[i] > regionHigh[i] ) { // here low[i] is C[i]
	  if ( b > squareradius ) return 0; // outside each other
	  dmin += b;
	  flag = true;
	}
	else { // C[i] is in between regionLow[i] and regionHigh[i]
	  //	  if ( min( a, b ) > squareradius ) return 3; // sphere inside box
	  if ( min( a, b ) <= squareradius ) flag = true;
	} 
      }
      if ( flag ) {
	if ( dmax < squareradius ) return 2; // box inside sphere
	else if ( dmin <= squareradius && squareradius <= dmax ) return 1; // intersect
	else return 0;// outside each other 
      }
      else return 3;// sphere inside box
    }
  }
  
  // add all points in subtree to 'found'
  void add( TreeNode< Data* >* ptr,  list< TPoint* >& found ) {
    if( ptr != 0 ) {
      if(  ptr->data->getPoint()  != 0 )
	found.push_back( ptr->data->getPoint() ); //only point in leaf node could be added
      add( ptr->leftPtr, found );
      add( ptr->rightPtr, found );
    }
  }
  
  // search k nearest neighbor
  void searchKDTree( TreeNode< Data* >* ptr,  const TPoint& qpoint, const int k, int depth ) {
    int dim, inter;
    TPoint* point;
    TPointElem split, temp, squareradius;
    pair< TPoint*, TPointElem > neighbor;
    
    if ( ptr->leftPtr == 0  &&  ptr->rightPtr == 0 ) { // leaf node
      point = ptr->data->getPoint();
      if ( knn.size() < k ) { // add this leaf node so far encountered
	neighbor.first = point;
	neighbor.second = squareDistance( qpoint, *point );
	knn.push( neighbor );
      }
      else { // check and update
	squareradius = knn.top().second;
	if ( intersect( 1, *point, *point, qpoint, qpoint, squareradius ) == 2 ) {// point is inside the sphere
	  knn.pop();
	  neighbor.first = point;
	  neighbor.second = squareDistance( qpoint, *point );
	  knn.push( neighbor ); // update
	}
	squareradius = knn.top().second;
	if ( intersect( 1, lowB, highB, qpoint, qpoint, squareradius ) == 3 ) stop = true;
	return;
      }
      return;
    }
    dim = depth % dimensions;
    split = ptr->data->getSplit();
    if ( qpoint[dim] <= split ) {
      temp = highB[dim];
      highB.set( dim, split );
      searchKDTree( ptr->leftPtr, qpoint, k, depth+1 );
      highB.set( dim, temp ); // restore bound
      if ( stop ) return; // stop recursion
    }
    else {
      temp = lowB[dim];
      lowB.set( dim, split);
      searchKDTree( ptr->rightPtr, qpoint, k, depth+1 );
      lowB.set( dim, temp ); // restore bound
      if ( stop ) return; // stop recursion
    }
    /*TPoint  tmppnt = *(knn.top().first);
    TPointElem tmprad = knn.top().second;
    cout << " tmp nearest neighbor " << tmppnt[0] << " " << tmppnt[1] << " " << tmppnt[2] << endl;
    cout << " tmp nearest radius " << tmprad << endl;*/
    //recursive call on farther son, if necessary
    if ( qpoint[dim] <= split ) {
      temp = lowB[dim];
      lowB.set( dim, split );
      squareradius = knn.top().second;
      inter = intersect( 1, lowB, highB, qpoint, qpoint, squareradius );
      if ( inter == 1 || inter ==2 ) searchKDTree( ptr->rightPtr, qpoint, k, depth+1 );
      lowB.set( dim, temp ); //restore bound
      if ( stop ) return; // stop recursion
    }
    else {
      temp = highB[dim];
      highB.set( dim, split );
      squareradius = knn.top().second;
      inter = intersect( 1, lowB, highB, qpoint, qpoint, squareradius );
      if ( inter == 1 || inter == 2 ) searchKDTree ( ptr->leftPtr, qpoint, k, depth+1 );
      highB.set( dim, temp ); // restore bound
      if ( stop ) return; // stop recursion
    }
    // see if we should return or terminate
    squareradius = knn.top().second;
    if (  intersect( 1, lowB, highB, qpoint, qpoint, squareradius ) == 3 ) stop = true;
    return;
  }
  
  TPointElem squareDistance( const TPoint& p0, const TPoint& p1 ) {
    TPointElem distance = 0;
    int i;
    for( i = 0; i < dimensions; i++ ) distance += pow( p0[i] - p1[i], 2.0 );
    return distance;
  }

  class comparenn {
  public:
    bool operator() ( pair< TPoint*, TPointElem > n0, pair< TPoint*, TPointElem > n1 ) {
      return n0.second > n1.second;
    }
  };
  priority_queue< pair< TPoint*, TPointElem >,vector< pair< TPoint*, TPointElem > >, comparenn > knn;
  TPoint lowB, highB;
  bool stop; // a flag indicating the recursive call should be stoped under a specific condition;
};

}

#endif


