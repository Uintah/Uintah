#include <SCICore/Malloc/Allocator.h>
#include <vector>
#include <list>
#include <iostream>
#include <assert.h>
#include <limits.h>

using namespace std;

/**************************************

CLASS

   RangeTree

   A template class for doing fast range queries on d-dimensional
   Points of any type.  
  
GENERAL INFORMATION
  
   RangeTree.h

   Wayne Witzel
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   RangeTree

DESCRIPTION
   If n is the number of points and d is the number of dimensions,
   building the tree (in the constructor) takes O(n * [log(n)]^(d-1))
   time and the structure uses the same order of space.  Queries
   take O([log(n)]^(d-1) + k) time where k is the number of elements
   in the query range to be reported.

   The Point class must define an operator[] method which gives the
   PointElem value in the ith dimension.  This must be valid for
   i = 0..(num_dimensions - 1).  The PointElement must define
   operators for <, <=, >, and >=.

   The tree structure is actually a multi-layer tree -- one layer
   per dimensions.  Each layer has a binary tree structure with
   nodes that each refer to "associated" trees on a lower layer
   that contain the subset of points represented at that node
   (all the points below the subtree of that node).  For a given
   node, v, it represents a set of point P(v).  It has an
   associated structure A(v) which represents the same set of
   points but considering one less dimension (in a lower layer).
   The node v has two children, lc(v) and rc(v) at the same
   layer, each representing subsets, P(lc(v)) and P(rc(v)), of
   P(v).

   The bottom layer (d == 0 layer) is different in that it is
   simply a sorted array form of a binary tree and it uses
   fractional cascading to speed up queries at that level.
   Fractional cascading uses an indexing system where each element
   in a bottom layer node, which I'll call A(v) where v is a node
   at the d == 1 layer), refer to the smallest elements in A(lc(v))
   and A(rc(v)) are >= to it.  Not that A(lc(v)) and A(rc(v)) contain
   subsets of the points in A(v) which is important to how fractional
   cascading works.  These indices are used during the query so
   that for each query at the d==1 layer, only a single binary search
   needs to be performed at the bottom layer (on the broadest set
   of points that may contain points in the query) after which it
   simply cascades down using the indices to know where in each
   array is the first element in the query.  This strategy reduces
   the time complexity of the query by a factor of log(n).

   This algorithm is taken from "Computation Geometry" by de Berg,
   van Kreveld, Overmars, and Schwarzkopf, in section 5.3.
   
**************************************/

template<class Point, class PointElem>
class RangeTree
{
public:
  // Build a range tree for the given set of points on the
  // given number of dimensions (must be >= 2).  There may
  // be Point's with equivalent values in the list but there
  // should not be Point*'s that are duplicated in the list
  // or this will cause an error.
  RangeTree(list<Point*> points, int dimensions /* >= 2 */);
  
  ~RangeTree();

  // Query for Point's in the given range such that
  // low[i] <= p[i] <= high[i] for each dimension.
  inline void query(Point& low, Point& high, list<Point*>& found);

  void topLevelDump(); // for debugging

  void bottomLevelDump(); // for debugging

private:
  class BaseLevelSet;

  // Node class for the tree's at every level except the bottom
  // d==0 level (which is a BaseLevelSet instead).
  struct RangeTreeNode
  {
    // RangeTreeNode constructor -- builds the tree recursively.
    // d is the dimension level (0..dimensions-1)
    // dSorted is a vector of Points sorted with respect to dimension d.
    // subDSorted is an array of sorted vectors, sorted with
    //   respect to each dimension up to d-1 respectfully
    // low and high give the range in the sorted vectors that
    //   this node is to deal with (prevents unnecessary duplication
    //   of the vectors.
    RangeTreeNode(int d, Point** dSorted, Point*** subDSorted,
		  int low, int high);

    void deleteStructure(int d);

    inline bool isLeaf()
    { return d_leftChild == NULL; }
    
    inline bool isPointInRange(Point& low, Point& high, int d);

    inline RangeTreeNode* getChild(int i)
    { return (i == 0) ? d_leftChild : d_rightChild; }

    inline RangeTreeNode* getOtherChild(int i)
    { return (i != 0) ? d_leftChild : d_rightChild; }
    
    void singleDimensionDump(int d);
    
    RangeTreeNode* d_leftChild;
    RangeTreeNode* d_rightChild;
    union {
      RangeTreeNode* rtn;
      BaseLevelSet* bls;
    } d_lowerLevel;
    Point* d_point; // either the leaf point or the sorting mid point
  };

  // Associated data structure for the base level (d == 0).
  // This contains a sorted array with links to subset arrays
  // for faster query (via fractional cascading) than can be
  // easily done at higher levels.
  class BaseLevelSet
  {
  public:
    BaseLevelSet(Point** points, int n);

    ~BaseLevelSet()
    { delete[] d_points; delete[] d_leftSubLinks; delete[] d_rightSubLinks; }
    
    inline void setLeftSubLinks(BaseLevelSet* leftSubset)
    { //d_leftSubset = leftSubset;
      setSubLinks(d_leftSubLinks, leftSubset); }
    
    inline void setRightSubLinks(BaseLevelSet* rightSubset)
    { //d_rightSubset = rightSubset;
      setSubLinks(d_rightSubLinks, rightSubset); }
    
    int findGreaterEq(PointElem e);

    // Use a lookup table to find the element in A(lc(v)) or A(rc(v))
    // (depending on whether 'left' is true or not) that is the first
    // one greater than or equal to the ith one in A(v). 
    inline int getSubGreaterEq(bool left, int i /* , BaseLevelSet* sub */
			                   /* pass in for verification */)
    { return left ? getLeftSubGreaterEq(i /* , sub */) :
                    getRightSubGreaterEq(i /* , sub */); }

    inline int getLeftSubGreaterEq(int i /*, BaseLevelSet* sub */)
    { /* assert(d_leftSubset == sub); */ return d_leftSubLinks[i]; }

    inline int getRightSubGreaterEq(int i /*, BaseLevelSet* sub */)
    { /* assert(d_rightSubset == sub); */ return d_rightSubLinks[i]; }

    // Append points to found list in the range from d_points[start]
    // until d_points[j] <= high.
    void getRange(list<Point*>& found, int start, PointElem high);

    void dump(); // for debugging
  private:
    void setSubLinks(int* sublinks, BaseLevelSet* subset);
		     
    Point** d_points; // array sorted in the zero'th dimension
    int d_size; // size of points array

    //BaseLevelSet* d_leftSubset; /* was used for debugging */
    //BaseLevelSet* d_rightSubset;
    
    // Indexes to the elements of A(lc(v)) and A(rc(v))
    // respectively that are greater than or equal to corresponding
    // elements in d_points.
    int* d_leftSubLinks;
    int* d_rightSubLinks;
  };

  void query(RangeTreeNode* root, list<Point*>& found,
	     Point& low, Point& high, int d);

  // helper for the above query method
  void queryFromSplit(RangeTreeNode* vsplit, list<Point*>& found,
		      Point& low, Point& high, int d);

  // helper for the above query method for d = 1 (just above
  // the base level)
  void queryFromSplitD1(RangeTreeNode* root, list<Point*>& found,
			Point& low, Point& high);
    
  RangeTreeNode* d_root;
  int d_dimensions;
};

template<class Point>
int comparePoints(Point* p1, Point* p2, int d)
{
  if ((*p1)[d] < (*p2)[d])
    return -1;
  else if ((*p1)[d] > (*p2)[d])
    return 1;
  else
    // Arbitrary pointer comparison to give a consistent ordering
    // when points are equal.
    return (int)(p1 - p2); 
}

template<class Point, class PointElem>
class CompareDimension
{
public:
  CompareDimension(int d)
    : d_dim(d) { }

  bool operator()(Point* x, Point* y)
  { return comparePoints(x, y, d_dim) < 0; }
private:
  int d_dim;
};

template<class Point, class PointElem>
RangeTree<Point, PointElem>::RangeTree(list<Point*> points, int dimensions)
  : d_dimensions(dimensions)
{
  assert(dimensions >= 2);
  
  // pre-sort in each dimension
  Point*** pointSorts = scinew Point**[dimensions];
  assert(points.size() <= INT_MAX);
  int n = (int)points.size();
  int i, j;
  list<Point*>::iterator iter;
  
  for (i = 0; i < dimensions; i++) {
    points.sort(CompareDimension<Point, PointElem>(i));
    pointSorts[i] = scinew Point*[n];
    for (iter = points.begin(), j = 0; iter != points.end(); iter++, j++)
      pointSorts[i][j] = *iter;
  }

  int d = dimensions - 1;
  d_root = scinew RangeTreeNode(d, pointSorts[d], pointSorts, 0, n);

  for (i = 0; i < d; i++)
    delete[] pointSorts[i];
  delete[] pointSorts;
}

template<class Point, class PointElem>
RangeTree<Point, PointElem>::~RangeTree()
{
  d_root->deleteStructure(d_dimensions - 1);
  delete d_root;
}

template<class Point, class PointElem>
void RangeTree<Point, PointElem>::query(Point& low, Point& high,
					list<Point*>& found)
{
  query(d_root, found, low, high, d_dimensions - 1);
}

template<class Point, class PointElem>
void RangeTree<Point, PointElem>::topLevelDump() // for debugging
{
  d_root->singleDimensionDump(d_dimensions - 1);
}

template<class Point, class PointElem>
void RangeTree<Point, PointElem>::bottomLevelDump()
{
  RangeTreeNode* rtn = d_root;
  for (int d = 2; d < d_dimensions; d++)
    rtn = rtn->d_lowerLevel.rtn;
  rtn->d_lowerLevel.bls->dump();
}

template<class Point>
bool testSorted(Point** sorted, int d, int low, int high)
{
  for (int i = low; i < high; i++)
    if ((*sorted[i])[d] >= (*sorted[i+1])[d])
      return false;
  return true;
}

template<class Point, class PointElem>
RangeTree<Point, PointElem>::RangeTreeNode::
RangeTreeNode(int d, Point** dSorted, Point*** subDSorted, int low, int high)
  : d_leftChild(NULL),
    d_rightChild(NULL),
    d_point(NULL)
{
  d_lowerLevel.rtn = NULL;
  
  int i, j;

  // build the associated sub tree for lower dimensions
  if (d > 1) {
    // Copy the sub-subDSorted arrays.
    // This is necessary, because in the process of building the
    // associated tree it will destroy the sorted order of these
    // sub-dimension sorted vectors.
    Point*** subSubDSorted = scinew Point**[d-1];
    for (i = 0; i < d-1; i++) {
      subSubDSorted[i] = scinew Point*[high - low];
      for (j = low; j < high; j++)
	subSubDSorted[i][j-low] = subDSorted[i][j];
    }
    d_lowerLevel.rtn = scinew RangeTreeNode(d-1, &subDSorted[d-1][low],
				  subSubDSorted, 0, high-low);
    for (i = 0; i < d-1; i++)
      delete[] subSubDSorted[i];
    delete[] subSubDSorted;
  }
  else if (d == 1) {
    d_lowerLevel.bls = scinew BaseLevelSet(&subDSorted[0][low], high-low);
  }

  // split points by the mid value of the (d-1)th dimension
  int mid_pos = (low + high) / 2;

  if (high - low > 1) {
    // contains more than just one point
    Point* p;
    d_point = dSorted[mid_pos];

    if (d > 0) {
      // split the sorted vectors of sub-dimensions between 'left'
      // and 'right' (two sides on dimension d -- left/right is figurative)
      Point** tmpLeftSorted = scinew Point*[mid_pos - low]; 
      Point** tmpRightSorted = scinew Point*[high - mid_pos];
      
      int left_index, right_index;
      
      for (i = 0; i < d; i++) {
	// split between the left sub-d sorted and the right
	// sub-d sorted according to mid (both sides will be
	// sorted because they were sorted befor splitting).
	left_index = right_index = 0;
	for (j = low; j < high; j++) {
	  p = subDSorted[i][j];
	  if (comparePoints(p, d_point, d) < 0)
	    tmpLeftSorted[left_index++] = p;
	  else
	    tmpRightSorted[right_index++] = p;
	}
	
	assert(left_index == mid_pos - low);
	assert(right_index == high - mid_pos);
	
	for (j = low; j < mid_pos; j++)
	  subDSorted[i][j] = tmpLeftSorted[j - low];
	for (j = mid_pos; j < high; j++)
	  subDSorted[i][j] = tmpRightSorted[j - mid_pos];
	
      }
      delete[] tmpLeftSorted;
      delete[] tmpRightSorted;
    }

    d_leftChild = scinew RangeTreeNode(d, dSorted, subDSorted, low, mid_pos);
    d_rightChild = scinew RangeTreeNode(d, dSorted, subDSorted, mid_pos,
					high); 

    if (d == 1) {
      d_lowerLevel.bls->setLeftSubLinks(d_leftChild->d_lowerLevel.bls);
      d_lowerLevel.bls->setRightSubLinks(d_rightChild->d_lowerLevel.bls);
    } 
  }
  else {
    d_point = dSorted[low];
  }
}

template<class Point, class PointElem>
void RangeTree<Point, PointElem>::RangeTreeNode::deleteStructure(int d)
{
  if (d > 1)
    d_lowerLevel.rtn->deleteStructure(d-1);
  else
    delete d_lowerLevel.bls;

  if (d_leftChild != NULL) {
    d_leftChild->deleteStructure(d);
    delete d_leftChild;
    //assert(d_rightChild != NULL)
    d_rightChild->deleteStructure(d);
    delete d_rightChild;
  }
}

template<class Point, class PointElem>
inline bool RangeTree<Point, PointElem>::RangeTreeNode::
isPointInRange(Point& low, Point& high, int d)
{
  for (int i = 0; i <= d; i++)
    if ((*d_point)[i] < low[i] || (*d_point)[i] > high[i])
      return false; // leaf point is out of range
  return true; // leaf point is in range
}

template<class Point, class PointElem>
void RangeTree<Point, PointElem>::RangeTreeNode::
singleDimensionDump(int d)
{
  if (d_leftChild != NULL) {
    assert(d_rightChild != NULL);
    d_leftChild->singleDimensionDump(d);
    d_rightChild->singleDimensionDump(d);
  }
  else
    cout << (*d_point)[d] << endl;
}

template<class Point, class PointElem>
RangeTree<Point, PointElem>::BaseLevelSet::
BaseLevelSet(Point** points, int n)
  : d_points(scinew Point*[n]), d_size(n),
    //d_leftSubset(NULL), d_rightSubset(NULL),
    d_leftSubLinks(scinew int[n+1]), d_rightSubLinks(scinew int[n+1])
{
  for (int i = 0; i < n; i++)
    d_points[i] = points[i];
}

template<class Point, class PointElem>
int RangeTree<Point, PointElem>::BaseLevelSet::
findGreaterEq(PointElem e) // binary searches
{
  int low = 0;
  int high = d_size;

  while (high - low > 1) {
    int mid = (high + low) / 2;
    if (e <= (*d_points[mid-1])[0])
      high = mid;
    else
      low = mid;
  }

  if (high - low == 0)
    return d_size;
  else {
    //assert((low == 0) || (e > (*d_points[low-1])[0]));
    return (e <= (*d_points[low])[0]) ? low : d_size;
  }
}

template<class Point, class PointElem>
void RangeTree<Point, PointElem>::BaseLevelSet::
setSubLinks(int* sublinks, BaseLevelSet* subset)
{
  int i;
  for (i = 0; i < d_size; i++)
    sublinks[i] = subset->findGreaterEq((*d_points[i])[0]);
  sublinks[i] = subset->d_size;
}

template<class Point, class PointElem>
void RangeTree<Point, PointElem>::BaseLevelSet::
getRange(list<Point*>& found, int start, PointElem high)
{
  for (int i = start; i < d_size; i++) {
    if ((*d_points[i])[0] <= high)
      found.push_back(d_points[i]);
    else
      return;
  }
}

template<class Point, class PointElem>
void RangeTree<Point, PointElem>::BaseLevelSet::
dump()
{
  for (int i = 0; i < d_size; i++) {
    cout << (*d_points[i])[0] << " "; //(" << d_points[i]->getId() << ") ";
  }
  cout << endl;
  if (d_leftSubset != NULL) {
    d_leftSubset->dump();
    d_rightSubset->dump();
  }
}


/* Actual query implementations below */

template<class Point, class PointElem>
void RangeTree<Point, PointElem>::
query(RangeTreeNode* root, list<Point*>& found,
      Point& low, Point& high, int d)
{
  RangeTreeNode* vsplit = root;
  if (root == NULL) return;

  // find split node which indicates the sub-tree at this
  // dimension level where the low..high points are contained
  while (!vsplit->isLeaf()) {
    if (low[d] > (*vsplit->d_point)[d])
      vsplit = vsplit->d_rightChild;
    else if (high[d] < (*vsplit->d_point)[d])
      vsplit = vsplit->d_leftChild;
    else 
      // split node found
      break;
  }

  if (vsplit->isLeaf()) {
    // if one child is NULL the other one should be and it is a leaf
    if (vsplit->isPointInRange(low, high, d))
      found.push_back(vsplit->d_point); // leaf point is in range
  }
  else {
    if (d > 1)
      queryFromSplit(vsplit, found, low, high, d);
    else
      queryFromSplitD1(vsplit, found, low, high);
  }
}

template<class Point, class PointElem>
void RangeTree<Point, PointElem>::
queryFromSplit(RangeTreeNode* vsplit, list<Point*>& found,
	       Point& low, Point& high, int d)
{
  // check left side (i = 0), then right side (i = 1)
  bool cond;
  for (int i = 0; i < 2; i++) {
    RangeTreeNode* v = vsplit->getChild(i);
    while (!v->isLeaf()) {
      if (i == 0)
	cond = (low[d] <= (*v->d_point)[d]); // left side check
      else
	cond = (high[d] >= (*v->d_point)[d]); // right side check

      if (cond) {
	query(v->getOtherChild(i)->d_lowerLevel.rtn, found, low, high, d-1);
	v = v->getChild(i);
      }
      else
	v = v->getOtherChild(i);
    }
    
    // check if v is in range itself
    if (v->isPointInRange(low, high, d))
      found.push_back(v->d_point); // leaf point is in range
  }
}

template<class Point, class PointElem>
void RangeTree<Point, PointElem>::
queryFromSplitD1(RangeTreeNode* vsplit, list<Point*>& found,
	       Point& low, Point& high)
{
  const int d = 1;
  BaseLevelSet* parentSub;
  int parentSubFirstGreaterEq;
  BaseLevelSet* sub;

  // Do one log(n) search here, then use constant lookup tables
  // to find firstGreaterEq indices as it descends down the tree.
  int splitSubFirstGreaterEq = vsplit->d_lowerLevel.bls->findGreaterEq(low[0]);
  int firstGreaterEqLow;

  // check left side (i = 0), then right side (i = 1)
  bool cond;
  for (int i = 0; i < 2; i++) {
    bool downLeftSide = (i == 0);
    bool wentLeft;
    RangeTreeNode* v = vsplit->getChild(i);
    parentSub = v->d_lowerLevel.bls;
    parentSubFirstGreaterEq = vsplit->
      d_lowerLevel.bls->getSubGreaterEq(downLeftSide, splitSubFirstGreaterEq
					/* , v->d_lowerLevel.bls */);
    
    while (!v->isLeaf()) {
      if (downLeftSide)
	cond = (low[d] <= (*v->d_point)[d]); // left side check
      else
	cond = (high[d] >= (*v->d_point)[d]); // right side check

      if (cond) {
	// constant time lookup (key to algorithm speedup)
	sub = v->getOtherChild(i)->d_lowerLevel.bls;
	firstGreaterEqLow = parentSub->
	  getSubGreaterEq(!downLeftSide, parentSubFirstGreaterEq /*, sub*/);
	sub->getRange(found, firstGreaterEqLow, high[0]);
	v = v->getChild(i);
	wentLeft = downLeftSide;
      }
      else {
	v = v->getOtherChild(i);
	wentLeft = !downLeftSide;
      }

      // constant time lookup (key to algorithm speedup)
      parentSubFirstGreaterEq = parentSub->
	getSubGreaterEq(wentLeft, parentSubFirstGreaterEq /* ,
			v->d_lowerLevel.bls */);

      parentSub = v->d_lowerLevel.bls;
    }
    
    // check if v is in range itself
    if (v->isPointInRange(low, high, d))
      found.push_back(v->d_point); // leaf point is in range
  }
}



