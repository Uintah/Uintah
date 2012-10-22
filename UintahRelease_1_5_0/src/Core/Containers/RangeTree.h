/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
#ifndef Core_Containers_RangeTree_h
#define Core_Containers_RangeTree_h

#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include <list>
#include <iostream>
#include <climits>
#include <cmath>

namespace SCIRun {
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
  
   
KEYWORDS
   RangeTree, range, query, nearest

DESCRIPTION
   If n is the number of points and d is the number of dimensions,
   building the tree (in the constructor) takes O(n * [log(n)]^(d-1))
   time and the structure uses the same order of space.  There are
   a number of queries that can be made on this RangeTree as described
   below.  Note that in order to use the nearest point queries, you
   must have the ALLOW_NEAREST_NEIGHBOR_QUERY template parameter be set
   to true, but this adds a significant overhead to construction time
   (about double in my experience) and storage space.

   Rectangular Query:
   query(const TPoint& low, const TPoint& high, list<TPoint*>& found) 
   O([log(n)]^(d-1) + k) time where k is the number of "found" points
   in the query range.  This is the most efficient type of query -- what
   the RangeTree is made for.

   Sphere Query:
   querySphere(const TPoint& p, TPointElem radius, list<TPoint*>& found)
   ... and variants
   O([log(n)]^(d-1) + k) time where k is at most that of a rectangular
   query encompassing the sphere and at the least the number of
   "found" points in the sphere.  The RangeTree is made for rectangular
   queries, but in a sphere query it prunes down the search space in
   toward the sphere from the encompassing rectancle it starts searching
   from.

   Nearest L1 Query:
   TPoint* queryNearestL1(const TPoint& p, TPointElem maximumL1Distance)
   Finds a point that is closer than or as close to p than any other point
   using the L1 metric to define distance.  The L1 metric distance is NOT
   the same as a geometric distance.  Where a geometric distance takes
   the sqrt of the sum of coordinate differences squared, the L1 distance
   takes the sum of the absolute coordinate difference.  In other words,
   the nearest L1 query gives a point on the smallest cube centered at
   p and containing query points (where the geometric nearest is the
   smallest sphere instead of cube).  This is done in O([log(n)]^(d-1)).

   Nearest Query:
   TPoint* queryNearest(const TPoint& p, TPointElem maximumL1Distance)
   Yields a point in the RangeTree that is nearest to p geometrically
   (a point on the sphere that is the smallest sphere centered at p that
   contains any points in the RangeTree).  This is done internally by
   finding the nearest L1 point, then using that distance from p to that
   point as the radius in a sphere query centered at p to find any points
   closer the the nearest L1 point.  This takes O([log(n)]^(d-1) + k)
   time where the k is that of the sphere query.


   Requirements:

   The TPoint class must implement a const version of operator[](int) that
   gives the TPointElem value in the ith dimension which must be valid for
   i = 0..(DIMENSIONS_ - 1).
   
   The TPointElem must define operators for <, <=, >, >=, ==, -=, +=, =0, *.


   Internals:

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
   array is the first element in the query.  This strategy reduces // 
   the time complexity of the query by a factor of log(n).

   The basic construction and range query algorithms were taken from
   "Computation Geometry" by de Berg, van Kreveld, Overmars, and
   Schwarzkopf, in section 5.3.
   
**************************************/

const int RANGE_LEFT = 0;
const int RANGE_RIGHT = 1;

template<class TPoint, class TPointElem,
  bool ALLOW_NEAREST_NEIGHBOR_QUERY=false>
class RangeTree
{
public:

  // Build a range tree for the given set of points on the
  // given number of dimensions (must be >= 2).  There may
  // be Point's with equivalent values in the list but there
  // should not be Point*'s that are duplicated in the list
  // or this will cause an error.
  RangeTree(std::list<TPoint*> points, int dimensions /* >= 2 */);
  
  ~RangeTree();

  // Query for Point's in the given range such that
  // low[i] <= p[i] <= high[i] for each dimension.
  inline void query(const TPoint& low, const TPoint& high,
		    std::list<TPoint*>& found)
  { query(root_, found, low, high, DIMENSIONS_ - 1); }

  // Query for Point's in the given range such that
  // distance(p, q[i]) <= radius (using L2, geometric definition of distance).
  inline void querySphere(const TPoint& p, TPointElem radius,
			  std::list<TPoint*>& found)
  { querySphereR2(p, radius*radius, found); }

  // given the radius squared instead of the radius itself
  inline void querySphereR2(const TPoint& p, TPointElem radiusSquared,
                            std::list<TPoint*>& found)
  { querySphere<true>(root_, found, p, radiusSquared, radiusSquared,
		      DIMENSIONS_ - 1); }


  // The non-strict version may include points outside the sphere in order
  // to speed up the query by not having to test candidate points to see
  // if they are indeed inside.  Note that this may even result in more
  // points than a query over the encompassing cube because it will include
  // leaf points that aren't checked (whereas a standarde range query does
  // check these).
  inline void querySphereNonStrict(const TPoint& p, TPointElem radius,
				   std::list<TPoint*>& found)
  { querySphereNonStrictR2(p, radius*radius, found); }

  // given the radius squared instead of the radius itself
  inline void querySphereNonStrictR2(const TPoint& p,
				     TPointElem radiusSquared,
				     std::list<TPoint*>& found)
  { querySphere<false>(root_, found, p, radiusSquared, radiusSquared,
		       DIMENSIONS_ - 1); }
  
  // Get the nearest point to a given point according to the usual
  // geometric L2 metric (sum of squares).
  // Note: If MAX is the maximum allowed value of type PointElem, then
  // maximumL1Distance should not be greater than MAX - p[d] for any d.
  TPoint* queryNearest(const TPoint& p, TPointElem maximumL1Distance
		      /* point found must be less than maximumL1Distance
			 away using the L1 metric (not a type) */);
  
  // Get the nearest point to a given point according to the L1
  // metric (sum of coordinate differences without squaring).  This
  // is faster than queryNearest() that uses the geometric L2 metric
  // (which does this query then needs to search for any closer points).
  // Note: If MAX is the maximum allowed value of type PointElem, then
  // maximumL1Distance should not be greater than MAX - p[d] for any d.
  TPoint* queryNearestL1(const TPoint& p, TPointElem maximumL1Distance
			/* point found must be less than maximumL1Distance
			   away using the L1 metric */);

  void topLevelDump(); // for debugging

  void bottomLevelDump(); // for debugging

  int getDimensions() const
  { return DIMENSIONS_; }
  
public:
  /* really only for internal use, although it's public */

  int** getDiagonalDirections()
  { return diagonalDirections_; }

  int getNumDiagDirections() const
  { return numDiagDirections_; }

  static inline TPointElem getL1MetricLength(const TPoint* p, int* direction,
					     int dimensions);

  static inline
  TPointElem getDistanceSquared(const TPoint& p1, const TPoint& p2,
				int dimensions);
  
  static inline TPointElem squared(TPointElem p)
  { return p * p; }  
private:
  class BaseLevelSet;

  // Node class for the tree's at every level except the bottom
  // d==0 level (which is a BaseLevelSet instead).
  class RangeTreeNode
  {
  public:
    // RangeTreeNode constructor -- builds the tree recursively.
    // d is the dimension level (0..dimensions-1)
    // dSorted is a vector of Points sorted with respect to dimension d.
    // subDSorted is an array of sorted vectors, sorted with
    //   respect to each dimension up to d-1 respectfully
    // low and high give the range in the sorted vectors that
    //   this node is to deal with (prevents unnecessary duplication
    //   of the vectors.
    RangeTreeNode(int d, TPoint** dSorted, TPoint*** subDSorted,
		  int low, int high, RangeTree* entireTree);

    static void deleteStructure(RangeTreeNode* node, int d);

    inline bool isLeaf()
    { return leftChild_ == NULL; }

    // returns true iff point_ is in the range between low and high, only
    // checking up to dimension d (asserting that it has already been
    // checked for the other dimension)s.
    inline bool isPointInRange(const TPoint& low, const TPoint& high, int d);

    // returns true iff point_ is in the given direction from p, checking
    // up to dimension d (asserting that it has already been checked for the
    // other dimensions).
    bool isPointInDirection(const TPoint& p, int* direction, int d);
    void singleDimensionDump(int d);
    
    RangeTreeNode* leftChild_;
    RangeTreeNode* rightChild_;
    union {
      RangeTreeNode* rtn;
      BaseLevelSet* bls;
    } lowerLevel_;
    TPoint* point_; // either the leaf point or the sorting mid point
  protected:
    ~RangeTreeNode() {}
  };

  // Associated data structure for the base level (d == 0).
  // This contains a sorted array with links to subset arrays
  // for faster query (via fractional cascading) than can be
  // easily done at higher levels.
  class BaseLevelSet
  {
  public:
    BaseLevelSet(TPoint** points, int n);
    ~BaseLevelSet();
    
    inline void setLeftSubLinks(BaseLevelSet* leftSubset)
    {
#if SCI_ASSERTION_LEVEL >= 2      
      leftSubset_ = leftSubset; // for debugging only
#endif
      setSubLinks(leftSubLinks_, leftSubset); }
    
    inline void setRightSubLinks(BaseLevelSet* rightSubset)
    {
#if SCI_ASSERTION_LEVEL >= 2      
      rightSubset_ = rightSubset; // for debugging only
#endif
      setSubLinks(rightSubLinks_, rightSubset); }

    template <int BOUND_FROM_SIDE>
    inline int findFirstFromSide(TPointElem e)
    { return (BOUND_FROM_SIDE == RANGE_LEFT) ?
	findFirstGreaterEq(e) : findLastLesserEq(e); }
    
    int findFirstGreaterEq(TPointElem e);
    inline int findLastLesserEq(TPointElem e)
    {
      int firstGrEq = findFirstGreaterEq(e);
      return (((firstGrEq < size_) && (*points_[firstGrEq])[0] == e) ?
	      firstGrEq : firstGrEq - 1);
    }
    
    // Use a lookup table to find the element in A(lc(v)) or A(rc(v))
    // (left or right versions) that is the first one greater than or
    // equal to the ith one in A(v).
    inline int getLeftSubGreaterEq(int i, BaseLevelSet* sub)
    {
      ASSERT(leftSubset_ == sub);
      CHECKARRAYBOUNDS(i, 0, size_+1);
      return leftSubLinks_[i];
    }
    inline int getRightSubGreaterEq(int i, BaseLevelSet* sub)
    {
      ASSERT(rightSubset_ == sub);
      CHECKARRAYBOUNDS(i, 0, size_+1);
      return rightSubLinks_[i];
    }

    // Append points to found list in the range from points_[start]
    // until points_[j] <= high.
    void getRange(std::list<TPoint*>& found, int start, TPointElem high);

    void setExtremePoints(int** diagonalDirections,
			  int numDiagDirections, int dimensions);

    inline TPoint* getExtremePointBoundLeft(int i, int directionIndex,
					    int numDiagDirections)
    { return getExtremePoint<RANGE_LEFT>(i,directionIndex,numDiagDirections); }

    inline TPoint* getExtremePointBoundRight(int i, int directionIndex,
					     int numDiagDirections)
    { return getExtremePoint<RANGE_RIGHT>(i,directionIndex,numDiagDirections);}
    
    template <int BOUND_FROM_SIDE>
    inline TPoint* getExtremePoint(int i, int directionIndex,
				  int numDiagDirections)
    {
      ASSERT((directionIndex < numDiagDirections/2) ==
	     (BOUND_FROM_SIDE == RANGE_RIGHT));
      if (i >= 0 && i < size_)
	return extremePoints_[i][directionIndex];
      else
	return 0; // a valid answer
    }

    inline TPoint* getPoint(int i)
    {
      CHECKARRAYBOUNDS(i, 0, size_);
      return points_[i];
    }

    int getSize() const
    { return size_; }
    
    void dump(); // for debugging
  private:
    BaseLevelSet(const BaseLevelSet&);
    BaseLevelSet& operator=(const BaseLevelSet&);
    
    TPoint** points_;
    int size_; // size of points_ array
    
    // Indexes to the elements of A(lc(v)) and A(rc(v))
    // respectively that are greater than or equal to corresponding
    // elements in points_. (for fractional cascading)
    int* leftSubLinks_;
    int* rightSubLinks_;
    
    // 2-dimensional array of dimensions [size_][2^dimensions].
    // Element [i][j] contains the least point in the jth diagonal
    // direction (using getL1MetricLength below) among all of the
    // pointElems_ from either 0 to i, for j < 2^d / 2, or i to size_-1,
    // for j >= 2^d / 2.  (Used for nearest neighbor search).
    TPoint*** extremePoints_;

    void setSubLinks(int* sublinks, BaseLevelSet* subset);

#if SCI_ASSERTION_LEVEL >= 2
    // for debugging use only
    BaseLevelSet* leftSubset_; 
    BaseLevelSet* rightSubset_;
#endif
  };
  
public:
  // The only reason this is not a member of RangeTreeNode is due to a
  // g++ compiler bug relating to template functions.
  template <int SIDE>
  static inline RangeTreeNode* getChild(RangeTreeNode* node)
  { return (SIDE == RANGE_LEFT) ? node->leftChild_ : node->rightChild_; }

  // This is necessare because of the same g++ compiler bug
  template <int BOUND_FROM_SIDE>
  inline TPoint* getExtremePoint(BaseLevelSet* bls, int i, int directionIndex,
				 int numDiagDirections)
  {
    return (BOUND_FROM_SIDE == RANGE_LEFT) ?
      bls->getExtremePointBoundLeft(i, directionIndex, numDiagDirections) :
      bls->getExtremePointBoundRight(i, directionIndex, numDiagDirections);
  }
  
  /* Likewise, these aren't in BaseLevelSet because of the same g++ compiler
     bug */
  
  // use cascading to get the index in the sub-BaseLevelSet (right or
  // left depending on SUB_SIDE) which corresponds to the first element >=
  // to element i in this set, or is the last element <= to element i in
  // this set (depending on BOUND_FROM_SIDE).
  template <int SUB_SIDE, int BOUND_FROM_SIDE>
  static inline int getCascadedSubIndex(BaseLevelSet* parent, int i,
					BaseLevelSet* sub);    

  template <int SIDE>
  static inline int getSubGreaterEq(BaseLevelSet* parent, int i,
				    BaseLevelSet* sub /* for verification */)
  {
    return (SIDE == RANGE_LEFT) ?
      parent->getLeftSubGreaterEq(i, sub) :
      parent->getRightSubGreaterEq(i, sub);
  }

private:
  // Traverse down one side of a tree in some dimension visiting each
  // node that is the highest node whose sorting point is contained within
  // some boundary in that dimension according to some BoundTester class.
  // (see BasicBoundTester and RadialBoundTester).
  template <int SIDE>
  class Traverser
  {
  public:
    Traverser(RangeTreeNode* vsplit, int d)
      : node_(vsplit), D_(d) { }
    Traverser(Traverser& traverser)
      : node_(traverser.node_), D_(traverser.D_) {}

    template <class BoundTester>
    inline bool goNext(const BoundTester& boundTester) {
      const int OTHER_SIDE = (SIDE + 1) % 2;
      ASSERT(!node_->isLeaf());
      
      node_ = getChild<SIDE>(node_);
      while (!node_->isLeaf() &&
	     !boundTester.isInBound((*node_->point_)[D_]))
      node_ = getChild<OTHER_SIDE>(node_);
      return !node_->isLeaf();
    }

    RangeTreeNode* getNode()
    { return node_; }

  protected:
    RangeTreeNode* node_;
    const int D_;
  private:
    Traverser& operator=(const Traverser&);    
  };

  // CascadeTraverser is a special Traverser (see above) that also
  // carries along fractional cascaded indices on a lower dimensional
  // level.  Meant for dimension 1 only, whose lower dimension is the base
  // dimension level 0.  Also, the CascadeTraverser might cut short where
  // the Traverser would have kept going in the case where the fractional
  // cascaded index is no longer in a valid range (meaning everything else
  // is out of range in the lowest dimension) if goNext is given a true
  // value for quitOnBadIndex or BOUND_FROM_SIDE == RANGE_RIGHT (in which
  // case quitOnBadIndex is ignored and it will quit regardless).
  template <int SIDE, int BOUND_FROM_SIDE>
  class CascadeTraverser : public Traverser<SIDE>
  {
  public:
    CascadeTraverser(RangeTreeNode* vsplit, int subIndex)
      : Traverser<SIDE>(vsplit, 1), subIndex_(subIndex)
    { ASSERT(subIndex != -1); }

    CascadeTraverser(CascadeTraverser& traverser)
      : Traverser<SIDE>(traverser), subIndex_(traverser.subIndex_) {}

    // quitOnBadIndex is only relevant if BOUND_FROM_SIDE == RANGE_LEFT,
    // otherwise it will quit regardless (because it can't handle negative
    // indices).
    template <class BoundTester>
    inline bool goNext(const BoundTester& boundTester,
		       bool quitOnBadIndex = true) {
      const int OTHER_SIDE = (SIDE + 1) % 2;
      ASSERT(!this->node_->isLeaf());
      BaseLevelSet* parentSub = this->node_->lowerLevel_.bls;
      BaseLevelSet* sub;
  
      this->node_ = getChild<SIDE>(this->node_);
      sub = this->node_->lowerLevel_.bls;
      subIndex_ =
	getCascadedSubIndex<SIDE, BOUND_FROM_SIDE>(parentSub, subIndex_, sub);
      if (((BOUND_FROM_SIDE == RANGE_RIGHT) || quitOnBadIndex) &&
      !isValidIndex(subIndex_, sub)) {
	return false; // no more points in range at the base level
      }
  
      while (!this->node_->isLeaf() &&
	     !boundTester.isInBound((*this->node_->point_)[this->D_])){
	parentSub = sub;    
	this->node_ = getChild<OTHER_SIDE>(this->node_);
	sub = this->node_->lowerLevel_.bls;
	subIndex_ = getCascadedSubIndex<OTHER_SIDE, BOUND_FROM_SIDE>
	  (parentSub, subIndex_, sub);
	if (((BOUND_FROM_SIDE == RANGE_RIGHT) || quitOnBadIndex) &&
	    !isValidIndex(subIndex_, sub)) {
	  return false;// no more points in range at the base level
	}
      }

      return !this->node_->isLeaf();
    }

    int getCurrentCascadedIndex()
    { return subIndex_; }

  private:        
    inline bool isValidIndex(int subIndex, const BaseLevelSet* sub)
    {
      if (BOUND_FROM_SIDE == RANGE_LEFT)
	return subIndex < sub->getSize();
      else
	return subIndex >= 0;
    }

    CascadeTraverser& operator=(const CascadeTraverser&);    
    int subIndex_;
  };

  /* BoundTester classes to be used in the goNext methods of the traversers */
  template <int SIDE>
  class BasicBoundTester
  {
  public:
    BasicBoundTester(const TPointElem& minmax)
      : minmax_(minmax) { }
    
    inline bool isInBound(TPointElem x) const
    { return (SIDE == RANGE_LEFT) ? x >= minmax_ : x <= minmax_; }
  private:
    TPointElem minmax_;
  };

  template <int SIDE>
  class RadialBoundTester
  {
  public:
    RadialBoundTester(const TPointElem& p, const TPointElem& radiusSquared)
      : p_(p), radiusSquared_(radiusSquared) { }

    inline bool isInBound(TPointElem x) const
    {
      TPointElem diff = x - p_;
      bool onOtherSide = ((SIDE == RANGE_LEFT) == (diff > 0));
      return onOtherSide || (diff * diff <= radiusSquared_);
    }
  private:
    TPointElem p_;
    TPointElem radiusSquared_;
  };

  void query(RangeTreeNode* root, std::list<TPoint*>& found,
	     const TPoint& low, const TPoint& high, int d);

  RangeTreeNode* findSplit(RangeTreeNode* root, TPointElem low,
			   TPointElem high, int d);
  RangeTreeNode* findSplitRadius2(RangeTreeNode* root, TPointElem p,
				  TPointElem radius2, int d);
  
  // template a version for the left side and a version for the right side
  template <int SIDE>
  void queryFromSplit(RangeTreeNode* vsplit, std::list<TPoint*>& found,
		      const TPoint& low, const TPoint& high, int d);

  // template a version for the left side and a version for the right side
  // for d == 1 case.
  template<int SIDE>
  void queryFromSplitD1(RangeTreeNode* vsplit, std::list<TPoint*>& found,
			const TPoint& low, const TPoint& high,
			int splitSubFirstGreaterEq);

  // query for the point in the sub-tree nearest to p, asserting that
  // all points in the sub-tree will be at least minL1Distance away (because
  // of information known about higher dimensions) which is used to stop the
  // search when the search in this dimension has gone farther than
  // nearestKnownDistance - minL1Distance away from the point.
  void queryNearestL1(RangeTreeNode* root, const TPoint& p,
		      const TPointElem& pL1Length, int d, int directionIndex,
		      const TPointElem& minL1Distance,
		      TPointElem& nearestKnownL1Distance, TPoint*& nearest);
  
  // templated according to which direction (left or right) is closer to
  // the p from vsplit in dimension d.
  template <int NEAR_SIDE, bool D1, int BASE_BOUND_SIDE /* only used if
							   D1 is true */>
  void queryNearestL1FromSplit(RangeTreeNode* vsplit, const TPoint& p,
			       const TPointElem& pL1Length, int d,
			       int directionIndex,
			       const TPointElem& minL1Distance,
			       TPointElem& nearestKnownL1Distance,
			       TPoint*& nearest);
  
  // Used by queryNearest methods to check if a candidate point is nearer
  // than the previously known nearest, if so then replace it.
  void setNearest(TPoint*& nearest, TPoint* candidate, int directionIndex,
		  const TPointElem& pL1Length,
		  TPointElem& nearestKnownL1Distance);

  template <bool STRICTLY_INSIDE>
  void querySphere(RangeTreeNode* root, std::list<TPoint*>& found, const TPoint& p,
		   TPointElem radiusSquared, TPointElem availableRadiusSquared,
		   int d);

  // query on one side in a dimension
  template <int SIDE, bool STRICTLY_INSIDE>
  void querySphereFromSplit(RangeTreeNode* vsplit, std::list<TPoint*>& found,
				const TPoint& p, TPointElem radiusSquared,
				TPointElem availableRadiusSquared, int d);
  // d == 1 version
  template<int SIDE, bool STRICTLY_INSIDE>
  void querySphereFromSplitD1(RangeTreeNode* vsplit, std::list<TPoint*>& found,
			      const TPoint& p, TPointElem radiusSquared,
			      TPointElem availableRadiusSquared,
			      int splitSubIndexP);
  
  template <int NEAR_SIDE>
  static inline TPointElem distance(const TPointElem& p, const TPointElem& x)
  { return (NEAR_SIDE == RANGE_LEFT) ? x - p : p - x; } 

  template <bool SIDE>
  static inline TPointElem addInDirection(const TPointElem& p,
					 const TPointElem& d)
  { return (SIDE == RANGE_LEFT) ? p - d : p + d; }

  
  static inline TPointElem getMinInRange(TPointElem x1, TPointElem x2)
  {
    if (x1 > 0)
      return (x2 > 0) ? ((x1 < x2) ? x1 : x2) /* both > 0 */: 0;
    else
      return (x2 > 0) ? 0 : ((x1 > x2) ? x1 : x2) /* both < 0 */;
  }

  void setDiagonalDirections();
  
  RangeTreeNode* root_;
  const int DIMENSIONS_;

  // Array of [2^DIMENSIONS_] X [DIMENSIONS_] giving all permutations
  // of points with elements either 1 or -1 in an order that should be
  // kept consistent.
  int** diagonalDirections_;
  int numDiagDirections_; // 2^DIMENSIONS_

  // The largest possible L1 distance between any two points in the set
  // (guaranteed to be >= to the actual largest distance).
  TPointElem largestPossibleL1Distance_;
};

template<class TPoint>
int comparePoints(TPoint* p1, TPoint* p2, int d)
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

template<class TPoint, class TPointElem>
class CompareDimension
{
public:
  CompareDimension(int d)
    : dim_(d) { }

  bool operator()(TPoint* x, TPoint* y)
  { return comparePoints(x, y, dim_) < 0; }
private:
  int dim_;
};

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
RangeTree(std::list<TPoint*> points, int dimensions)
  : DIMENSIONS_(dimensions), diagonalDirections_(0), numDiagDirections_(0)
{
  ASSERT(dimensions >= 2);

  if (ALLOW_NEAREST_NEIGHBOR_QUERY)
    // only used when needing to do nearest neighbor query
    setDiagonalDirections(); 
  
  // pre-sort in each dimension
  TPoint*** pointSorts = scinew TPoint**[dimensions];
  ASSERT(points.size() <= INT_MAX);
  int n = (int)points.size();
  int i, j;
  typename std::list<TPoint*>::iterator iter;

  largestPossibleL1Distance_ = 0;
  for (i = 0; i < dimensions; i++) {
    points.sort(CompareDimension<TPoint, TPointElem>(i));
    pointSorts[i] = scinew TPoint*[n];
    for (iter = points.begin(), j = 0; iter != points.end(); iter++, j++)
      pointSorts[i][j] = *iter;

    largestPossibleL1Distance_ +=
      (*pointSorts[i][j-1])[i] - (*pointSorts[i][0])[i];
  }

  int d = dimensions - 1;
  root_ = scinew RangeTreeNode(d, pointSorts[d], pointSorts, 0, n, this);

  for (i = 0; i < dimensions; i++)
    delete[] pointSorts[i];
  delete[] pointSorts;
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::~RangeTree()
{
  RangeTreeNode::deleteStructure(root_, DIMENSIONS_ - 1);
  if (ALLOW_NEAREST_NEIGHBOR_QUERY) {
    for (int i = 0; i < numDiagDirections_; i++)
      delete diagonalDirections_[i];
    delete diagonalDirections_;
  }
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
TPoint* RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
queryNearest(const TPoint& p, TPointElem maximumValue)
{
  // Get the nearestL1 point to narrow down the search
  ASSERTL1(ALLOW_NEAREST_NEIGHBOR_QUERY);
  TPoint* nearest;
  if (ALLOW_NEAREST_NEIGHBOR_QUERY) { // don't compile this if false
    TPoint* nearestL1 = queryNearestL1(p, maximumValue);
    if (nearestL1 == 0) return 0;
    TPointElem distSquared = getDistanceSquared(p, *nearestL1, DIMENSIONS_);
    if (distSquared == 0)
      return nearestL1; // can't get closer than zero    
    std::list<TPoint*> candidates;
    querySphereNonStrictR2(p, distSquared, candidates);
    
    nearest = nearestL1;
    TPointElem nearestDistSquared = distSquared;
    // just do a linear search on the results (which hopefully, and in
    // most cases, will be a relatively small subset of points).
    for (typename std::list<TPoint*>::iterator it = candidates.begin();
	 it != candidates.end(); it++) {
      distSquared = getDistanceSquared(p, **it, DIMENSIONS_);
      if (distSquared < nearestDistSquared) {
	nearest = *it;
	nearestDistSquared = distSquared;
      }
    }
  }
  return nearest;
}


template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
TPoint* RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
queryNearestL1(const TPoint& p, TPointElem maximumValue)
{
  ASSERTL1(ALLOW_NEAREST_NEIGHBOR_QUERY);
  TPoint* nearest = 0;  
  if (ALLOW_NEAREST_NEIGHBOR_QUERY) { // don't compile this if false
    TPointElem nearestKnownL1Distance = maximumValue;
    
    // find the closest by looking for the closest in each direction and
    // keeping the closest one of all.
    for (int directionIndex = 0; directionIndex < numDiagDirections_;
       directionIndex++) {
      TPointElem pL1Length =
	getL1MetricLength(&p, diagonalDirections_[directionIndex],
			  DIMENSIONS_);
      queryNearestL1(root_, p, pL1Length, DIMENSIONS_ - 1, directionIndex, 0,
		     nearestKnownL1Distance, nearest);
      if (nearestKnownL1Distance == 0)
	break; // can't get closer than zero
    }
  }
  return nearest;
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
topLevelDump() // for debugging
{
  root_->singleDimensionDump(DIMENSIONS_ - 1);
  std::cout << std::endl;
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
bottomLevelDump()
{
  RangeTreeNode* rtn = root_;
  for (int d = 2; d < DIMENSIONS_; d++)
    rtn = rtn->lowerLevel_.rtn;
  rtn->lowerLevel_.bls->dump();
}

template<class TPoint>
bool testSorted(TPoint** sorted, int d, int low, int high)
{
  for (int i = low; i < high; i++)
    if ((*sorted[i])[d] >= (*sorted[i+1])[d])
      return false;
  return true;
}


template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::RangeTreeNode::
RangeTreeNode(int d, TPoint** dSorted, TPoint*** subDSorted, int low, int high,
	      RangeTree* entireTree)
  : leftChild_(NULL),
    rightChild_(NULL),
    point_(NULL)
{
  lowerLevel_.rtn = NULL;
  int i, j;

  // build the associated sub tree for lower dimensions
  if (d > 1) {
    // Copy the sub-subDSorted arrays.
    // This is necessary, because in the process of building the
    // associated tree it will destroy the sorted order of these
    // sub-dimension sorted vectors.
    TPoint*** subSubDSorted = scinew TPoint**[d-1];
    for (i = 0; i < d-1; i++) {
      subSubDSorted[i] = scinew TPoint*[high - low];
      for (j = low; j < high; j++)
	subSubDSorted[i][j-low] = subDSorted[i][j];
    }
    lowerLevel_.rtn =
      scinew RangeTreeNode(d-1, &subDSorted[d-1][low], subSubDSorted, 0,
			   high-low, entireTree);
    for (i = 0; i < d-1; i++)
      delete[] subSubDSorted[i];
    delete[] subSubDSorted;
  }
  else if (d == 1) {
    lowerLevel_.bls = scinew BaseLevelSet(&subDSorted[0][low], high-low);
    if (ALLOW_NEAREST_NEIGHBOR_QUERY)
      lowerLevel_.bls->setExtremePoints(entireTree->getDiagonalDirections(),
					entireTree->getNumDiagDirections(),
					entireTree->getDimensions());
  }

  // split points by the mid value of the (d-1)th dimension
  int mid_pos = (low + high) / 2;

  if (high - low > 1) {
    // contains more than just one point
    TPoint* p;
    point_ = dSorted[mid_pos];

    if (d > 0) {
      // split the sorted vectors of sub-dimensions between 'left'
      // and 'right' (two sides on dimension d -- left/right is figurative)
      TPoint** tmpLeftSorted = scinew TPoint*[mid_pos - low]; 
      TPoint** tmpRightSorted = scinew TPoint*[high - mid_pos];
      
      int left_index, right_index;
      
      for (i = 0; i < d; i++) {
	// split between the left sub-d sorted and the right
	// sub-d sorted according to mid (both sides will be
	// sorted because they were sorted befor splitting).
	left_index = right_index = 0;
	for (j = low; j < high; j++) {
	  p = subDSorted[i][j];
	  if (comparePoints(p, point_, d) < 0)
	    tmpLeftSorted[left_index++] = p;
	  else
	    tmpRightSorted[right_index++] = p;
	}
	
	ASSERT(left_index == mid_pos - low);
	ASSERT(right_index == high - mid_pos);
	
	for (j = low; j < mid_pos; j++)
	  subDSorted[i][j] = tmpLeftSorted[j - low];
	for (j = mid_pos; j < high; j++)
	  subDSorted[i][j] = tmpRightSorted[j - mid_pos];
	
      }
      delete[] tmpLeftSorted;
      delete[] tmpRightSorted;
    }

    leftChild_ = scinew RangeTreeNode(d, dSorted, subDSorted, low, mid_pos,
				      entireTree);
    rightChild_ = scinew RangeTreeNode(d, dSorted, subDSorted, mid_pos,
				       high, entireTree); 

    if (d == 1) {
      lowerLevel_.bls->setLeftSubLinks(leftChild_->lowerLevel_.bls);
      lowerLevel_.bls->setRightSubLinks(rightChild_->lowerLevel_.bls);
    } 
  }
  else {
    point_ = dSorted[low];
  }
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::RangeTreeNode::
deleteStructure(RangeTreeNode* node, int d)
{
  if (d > 1) {
    deleteStructure(node->lowerLevel_.rtn, d-1);
  }
  else
    delete node->lowerLevel_.bls;

  if (node->leftChild_ != NULL) {
    deleteStructure(node->leftChild_, d);
    ASSERT(node->rightChild_ != NULL)
    deleteStructure(node->rightChild_, d);
  }
  
  delete node;
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
inline bool
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::RangeTreeNode::
isPointInRange(const TPoint& low, const TPoint& high, int d)
{
  for (int i = 0; i <= d; i++)
    if ((*point_)[i] < low[i] || (*point_)[i] > high[i])
      return false; // leaf point is out of range
  return true; // leaf point is in range
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
bool
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::RangeTreeNode::
isPointInDirection(const TPoint& p, int* direction, int d)
{
  for (int i = 0; i <= d; i++) {
    if (direction[i] < 0) {
      if (p[i] < (*point_)[i])
	return false;
    }
    else {
      if (p[i] > (*point_)[i])
	return false;
    }
  }
  return true;
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::RangeTreeNode::
singleDimensionDump(int d)
{
  if (leftChild_ != NULL) {
    ASSERT(rightChild_ != NULL);
    std::cout << "(";
    leftChild_->singleDimensionDump(d);
    std::cout << ",{" << (*point_)[d] << "},";
    rightChild_->singleDimensionDump(d);
    std::cout << ")";
  }
  else
    std::cout /*<< point_->getId() << ":" */ << (*point_)[d];
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::BaseLevelSet::
BaseLevelSet(TPoint** points, int n)
  : points_(scinew TPoint*[n]), size_(n),
    leftSubLinks_(scinew int[n+1]), rightSubLinks_(scinew int[n+1]),
    extremePoints_(0)
{
  ASSERT(size_ >= 1);
  for (int i = 0; i < size_; i++)
    points_[i] = points[i];
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::BaseLevelSet::
~BaseLevelSet()
{
  delete[] points_;
  delete[] leftSubLinks_;
  delete[] rightSubLinks_;
  if (extremePoints_ != 0) {
    for (int i = 0; i < size_; i++)
      delete[] extremePoints_[i];
    delete[] extremePoints_;
  }
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::BaseLevelSet::
setExtremePoints(int** diagonalDirections, int numDiagDirections,
		 int dimensions)
{
  extremePoints_ = scinew TPoint**[size_];
  TPointElem L1length;
  TPointElem L1length2;
  int i;
  int dir;

  for (i = 0; i < size_; i++)
    extremePoints_[i] = scinew TPoint*[numDiagDirections];
  
  if (numDiagDirections > 0) {
    for (dir = 0; dir < numDiagDirections/2; dir++) {
      // These directions all point negative in the lowest dimension,
      // so these will be for queries in these directions which will always
      // be from -infinity in the lowest dimension (specifically when doing
      // a nearest neighbor search which is what this is for), and thus
      // we want the extreme point from the ith point to the left.
      ASSERT(diagonalDirections[dir][0] < 0);
      extremePoints_[0][dir] = points_[0];
      L1length = getL1MetricLength(points_[0], diagonalDirections[dir],
				   dimensions);
      for (i = 1; i < size_; i++) {
	L1length2 = getL1MetricLength(points_[i], diagonalDirections[dir],
				      dimensions);
	if (L1length2 < L1length) {
	  L1length = L1length2;
	  extremePoints_[i][dir] = points_[i];
	}
	else
	  extremePoints_[i][dir] = extremePoints_[i-1][dir];
      }
    }
    for (dir = numDiagDirections/2; dir < numDiagDirections; dir++) {
      // These directions all point positive in the lowest dimension,
      // so these will be for queries in these directions which will always
      // go to +infinity in the lowest dimension (specifically when doing
      // a nearest neighbor search which is what this is for), and thus
      // we want the extreme point from the ith point to the right.
      ASSERT(diagonalDirections[dir][0] > 0);
      extremePoints_[size_-1][dir] = points_[size_-1];
      L1length = getL1MetricLength(points_[size_-1], diagonalDirections[dir],
				   dimensions);
      for (i = size_-2; i >= 0; i--) {
	L1length2 = getL1MetricLength(points_[i], diagonalDirections[dir],
				      dimensions);
	if (L1length2 < L1length) {
	  L1length = L1length2;
	  extremePoints_[i][dir] = points_[i];
	}
	else
	  extremePoints_[i][dir] = extremePoints_[i+1][dir];
      }
    }
  }  
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
template<int SUB_SIDE, int BOUND_FROM_SIDE>
inline int
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
getCascadedSubIndex(BaseLevelSet* parent, int i, BaseLevelSet* sub)
{
  int subIndex = getSubGreaterEq<SUB_SIDE>(parent, i, sub);
  if ((BOUND_FROM_SIDE == RANGE_RIGHT) &&
      ((subIndex == sub->getSize()) ||
       ((*sub->getPoint(subIndex))[0] > (*parent->getPoint(i))[0])))
    return subIndex - 1; // go lesser instead of greater
  else
    return subIndex; // equal or greater
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
inline TPointElem RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
getL1MetricLength(const TPoint* p, int* direction, int dimensions)
{
  TPointElem result = 0;
  for (int d = 0; d < dimensions; d++) {
    if (direction[d] > 0)
      result += (*p)[d];
    else
      result -= (*p)[d];
  }
  return result;
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
inline TPointElem RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
getDistanceSquared(const TPoint& p1, const TPoint& p2, int dimensions)
{
  TPointElem result = 0;
  for (int d = 0; d < dimensions; d++) {
    TPointElem diff = p1[d] - p2[d];
    result += diff * diff;
  }
  return result;
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
int RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::BaseLevelSet::
findFirstGreaterEq(TPointElem e) // binary searches
{
  int low = 0;
  int high = size_;

  while (high - low > 1) {
    int mid = (high + low) / 2;
    if (e <= (*points_[mid-1])[0])
      high = mid;
    else
      low = mid;
  }

  if (high - low == 0)
    return size_;
  else {
    //ASSERT((low == 0) || (e > (*points_[low-1])[0]));
    return (e <= (*points_[low])[0]) ? low : size_;
  }
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::BaseLevelSet::
setSubLinks(int* sublinks, BaseLevelSet* subset)
{
  int i;
  for (i = 0; i < size_; i++)
    sublinks[i] = subset->findFirstGreaterEq((*points_[i])[0]);
  sublinks[i] = subset->size_;
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::BaseLevelSet::
getRange(std::list<TPoint*>& found, int start, TPointElem high)
{
  for (int i = start; i < size_; i++) {
    if ((*points_[i])[0] <= high)
      found.push_back(points_[i]);
    else
      return;
  }
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::BaseLevelSet::
dump()
{
  for (int i = 0; i < size_; i++) {
    std::cout << (*points_[i])[0] << " "; //(" << points_[i]->getId() << ") ";
  }
  std::cout << std::endl;
  if (this->leftSubset_ != NULL) {
    this->leftSubset_->dump();
    this->rightSubset_->dump();
  }
}


/* Traverser class methods -- for traversing down a tree */
/* Moved to class file - Steve */

/* Actual query implementations below */

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
typename RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::RangeTreeNode*
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
findSplit(RangeTreeNode* root, TPointElem low, TPointElem high, int d)
{
  RangeTreeNode* vsplit = root;

  // find split node which indicates the sub-tree at this
  // dimension level where the low..high points are contained
  while (!vsplit->isLeaf()) {
    if ((*vsplit->point_)[d] < low)
      vsplit = vsplit->rightChild_;
    else if ((*vsplit->point_)[d] > high)
      vsplit = vsplit->leftChild_;
    else 
      // split node found
      break;
  }

  return vsplit;
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
typename RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::RangeTreeNode*
RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
findSplitRadius2(RangeTreeNode* root, TPointElem p, TPointElem radius2, int d)
{
  RangeTreeNode* vsplit = root;

  // find split node which indicates the sub-tree at this
  // dimension level where the low..high points are contained
  while (!vsplit->isLeaf()) {
    TPointElem diff = (*vsplit->point_)[d] - p;
    TPointElem diff2 = diff*diff;
    if (diff2 > radius2)
      vsplit = (diff > 0) ? vsplit->leftChild_ : vsplit->rightChild_;
    else 
      // split node found
      break;
  }

  return vsplit;
}


/* range query */

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
query(RangeTreeNode* root, std::list<TPoint*>& found,
      const TPoint& low, const TPoint& high, int d)
{
  if (root == NULL)
    return;
  
  RangeTreeNode* vsplit = findSplit(root, low[d], high[d], d);

  if (vsplit->isLeaf()) {
    if (vsplit->isPointInRange(low, high, d))
      found.push_back(vsplit->point_); // leaf point is in range
  }
  else {
    if (d == 1) {
      // base level case.  d = 1 queries down to d = 0.
      
    // Do one log(n) search here, then use constant lookup tables
      // to find firstGreaterEq indices as it descends down the tree.
      int splitSubFirstGreaterEq =
	vsplit->lowerLevel_.bls->findFirstGreaterEq(low[0]);
      
      // traverse the left side
      queryFromSplitD1<RANGE_LEFT>(vsplit, found, low, high,
				       splitSubFirstGreaterEq);
      // traverse the right side
      queryFromSplitD1<RANGE_RIGHT>(vsplit, found, low, high,
					splitSubFirstGreaterEq);
    }
    else {
      // traverse the left side
      queryFromSplit<RANGE_LEFT>(vsplit, found, low, high, d);
      // traverse the right side
      queryFromSplit<RANGE_RIGHT>(vsplit, found, low, high, d);
    }
  }
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
template<int SIDE>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
queryFromSplit(RangeTreeNode* vsplit, std::list<TPoint*>& found,
	       const TPoint& low, const TPoint& high, int d)
{
  const int OTHER_SIDE = (SIDE + 1) % 2;
  TPointElem minmax = ((SIDE == RANGE_LEFT) ? low[d] : high[d]);
  BasicBoundTester<SIDE> boundTester(minmax);

  // traverse the tree and do queries on the inner sub-trees that are
  // fully contained in the range at this level
  Traverser<SIDE> traverser(vsplit, d);
  while (traverser.goNext(boundTester)) {
    query(getChild<OTHER_SIDE>(traverser.getNode())->lowerLevel_.rtn,
	  found, low, high, d-1);
  }

  // check the leaf point
  RangeTreeNode* leafNode = traverser.getNode();
  if (leafNode->isPointInRange(low, high, d))
    found.push_back(leafNode->point_); // leaf point is in range
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
template<int SIDE>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
queryFromSplitD1(RangeTreeNode* vsplit, std::list<TPoint*>& found,
		 const TPoint& low, const TPoint& high,
		 int splitSubFirstGreaterEq /* only used if d1 is true */)
{
  // This level uses fractional cascading
  const int d = 1;
  const int OTHER_SIDE = (SIDE + 1) % 2;
  
  TPointElem minmax = ((SIDE == RANGE_LEFT) ?
		      low[d] : high[d]);
  BasicBoundTester<SIDE> boundTester(minmax);

  // traverse the tree and do queries on the inner sub-trees that are
  // fully contained in the range at this level
  CascadeTraverser<SIDE, RANGE_LEFT> traverser(vsplit, splitSubFirstGreaterEq);
  while (traverser.goNext(boundTester)) {
    // base case, d == 1 querying d = 0
    RangeTreeNode* node = traverser.getNode();
    BaseLevelSet* sub = getChild<OTHER_SIDE>(node)->lowerLevel_.bls;
    // fractional cascading - constant time lookup
    int cascadedSubIndex = traverser.getCurrentCascadedIndex();

    int firstGreaterEqLow = getCascadedSubIndex<OTHER_SIDE, RANGE_LEFT>
      (node->lowerLevel_.bls, cascadedSubIndex, sub);
      
    sub->getRange(found, firstGreaterEqLow, high[0]);
  }

  // check the leaf point
  RangeTreeNode* leafNode = traverser.getNode();
  if (leafNode->isPointInRange(low, high, d))
    found.push_back(leafNode->point_); // leaf point is in range
}


/* Sphere range query */

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
template <bool STRICTLY_INSIDE>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
querySphere(RangeTreeNode* root, std::list<TPoint*>& found, const TPoint& p,
	    TPointElem radiusSquared, TPointElem availableRadiusSquared, int d)
{
  if (root == NULL)
    return;
  
  RangeTreeNode* vsplit = findSplitRadius2(root, p[d],
					   availableRadiusSquared, d);
	 
  if (vsplit->isLeaf()) {
   if (!STRICTLY_INSIDE ||
       getDistanceSquared(p, *vsplit->point_, DIMENSIONS_) <= radiusSquared)
      found.push_back(vsplit->point_); // leaf point is in range
  }
  else {
    if (d == 1) {
      // base level case.  d = 1 queries down to d = 0.
      
      // Do one log(n) search here, then use constant lookup tables
      // to find firstGreaterEq indices as it descends down the tree.
      int splitSubIndexP =
	vsplit->lowerLevel_.bls->findFirstGreaterEq(p[0]);
      
      // traverse the left side
      querySphereFromSplitD1<RANGE_LEFT, STRICTLY_INSIDE>
	(vsplit, found, p, radiusSquared, availableRadiusSquared,
	 splitSubIndexP);
      // traverse the right side
      querySphereFromSplitD1<RANGE_RIGHT, STRICTLY_INSIDE>
	(vsplit, found, p, radiusSquared, availableRadiusSquared,
	 splitSubIndexP);
    }
    else {
      // traverse the left side
      querySphereFromSplit<RANGE_LEFT, STRICTLY_INSIDE>
	(vsplit, found, p, radiusSquared, availableRadiusSquared, d);
      // traverse the right side
      querySphereFromSplit<RANGE_RIGHT, STRICTLY_INSIDE>
	(vsplit, found, p, radiusSquared, availableRadiusSquared, d);
    }
  } 
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
template<int SIDE, bool STRICTLY_INSIDE>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
querySphereFromSplit(RangeTreeNode* vsplit, std::list<TPoint*>& found,
		     const TPoint& p, TPointElem radiusSquared,
		     TPointElem availableRadiusSquared, int d)
{
  const int OTHER_SIDE = (SIDE + 1) % 2;
  RadialBoundTester<SIDE> boundTester(p[d], availableRadiusSquared);

  // traverse the tree and do queries on the inner sub-trees that are
  // fully contained in the range at this level
  Traverser<SIDE> traverser(vsplit, d);
  TPointElem prevDiff = (*traverser.getNode()->point_)[d] - p[d];
  TPointElem diff;
  TPointElem minDiff;
  while (traverser.goNext(boundTester)) {
    diff = (*traverser.getNode()->point_)[d] - p[d];

    // All query points below will be between p[d] + prevDiff and p[d] + diff,
    // get the closest possible value in that range in order to reduce the
    // availableRadius in the next level.
    minDiff = getMinInRange(diff, prevDiff);
    querySphere<STRICTLY_INSIDE>
      (getChild<OTHER_SIDE>(traverser.getNode())->lowerLevel_.rtn, found, p,
       radiusSquared, availableRadiusSquared - minDiff*minDiff, d-1);
    prevDiff = diff;
  }

  // check the leaf point
  RangeTreeNode* leafNode = traverser.getNode();
  if (!STRICTLY_INSIDE ||
      getDistanceSquared(p, *leafNode->point_, DIMENSIONS_) <= radiusSquared) {
    found.push_back(leafNode->point_); // leaf point is in range
  }
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
template<int SIDE, bool STRICTLY_INSIDE>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
querySphereFromSplitD1(RangeTreeNode* vsplit, std::list<TPoint*>& found,
		       const TPoint& p, TPointElem radiusSquared,
		       TPointElem availableRadiusSquared,
		       int splitSubIndexP)
{
  // This level uses fractional cascading
  const int d = 1;
  const int OTHER_SIDE = (SIDE + 1) % 2;
  int i;
  RadialBoundTester<SIDE> boundTester(p[d], availableRadiusSquared);

  // traverse the tree and do queries on the inner sub-trees that are
  // fully contained in the range at this level
  CascadeTraverser<SIDE, RANGE_LEFT> traverser(vsplit, splitSubIndexP);
  TPointElem prevDiff = (*traverser.getNode()->point_)[d] - p[d];
  TPointElem diff;
  TPointElem minDiff;
  while (traverser.goNext(boundTester, false /* don't quit on a bad index
					        because it needs to check
					        elements before it */)) {
    // base case, d == 1 querying d = 0
    RangeTreeNode* node = traverser.getNode();
    BaseLevelSet* sub = getChild<OTHER_SIDE>(node)->lowerLevel_.bls;
    // fractional cascading - constant time lookup
    int cascadedSubIndex = traverser.getCurrentCascadedIndex();

    int indexP = getCascadedSubIndex<OTHER_SIDE, RANGE_LEFT>
      (node->lowerLevel_.bls, cascadedSubIndex, sub);

    // All query points below will be between p[d] + prevDiff and p[d] + diff,
    // get the closest possible value in that range in order to reduce the
    // availableRadius in the next level.
    diff = (*traverser.getNode()->point_)[d] - p[d];
    minDiff = getMinInRange(diff, prevDiff);
    TPointElem curAvailRadius2 = availableRadiusSquared - minDiff*minDiff;
    prevDiff = diff;
    
    for (i = indexP; i < sub->getSize() &&
	   squared((*sub->getPoint(i))[0] - p[0]) <= curAvailRadius2; i++) {
      if (!STRICTLY_INSIDE ||
	  (getDistanceSquared(p, *sub->getPoint(i), DIMENSIONS_) <=
	   radiusSquared))
	found.push_back(sub->getPoint(i));
    }
    for (i = indexP-1; i >= 0 &&
	   squared(p[0] - (*sub->getPoint(i))[0]) <= curAvailRadius2; i--) {
      if (!STRICTLY_INSIDE ||
	  (getDistanceSquared(p, *sub->getPoint(i), DIMENSIONS_) <=
	   radiusSquared))
	found.push_back(sub->getPoint(i));
    }
  }

  // check the leaf point
  RangeTreeNode* leafNode = traverser.getNode();
  if (!STRICTLY_INSIDE ||
      getDistanceSquared(p, *leafNode->point_, DIMENSIONS_) <= radiusSquared) {
    found.push_back(leafNode->point_); // leaf point is in range
  }
}


/* Nearest neighbor query */

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
queryNearestL1(RangeTreeNode* root, const TPoint& p,
	       const TPointElem& pL1Length, int d, int directionIndex,
	       const TPointElem& minL1Distance,
	       TPointElem& nearestKnownL1Distance, TPoint*& nearest)
{
  if (root == NULL)
    return;
  
  bool posDirection = (diagonalDirections_[directionIndex][d] > 0);
  RangeTreeNode* vsplit;

  if (posDirection)
    vsplit = findSplit(root, p[d],
		       p[d] + (nearestKnownL1Distance - minL1Distance), d);
  else
    vsplit = findSplit(root, p[d] - (nearestKnownL1Distance - minL1Distance),
		       p[d], d);   

  if (vsplit->isLeaf()) {
    if (vsplit->isPointInDirection(p, diagonalDirections_[directionIndex], d)){
	setNearest(nearest, vsplit->point_, directionIndex, pL1Length,
		   nearestKnownL1Distance);
    }
	
    return;
  }
  else {
    // chose the correct templated method to call
    if (posDirection) {
      if (d > 1)
	queryNearestL1FromSplit<RANGE_LEFT, false, 0>
	  (vsplit, p, pL1Length, d, directionIndex, minL1Distance,
	   nearestKnownL1Distance, nearest);
      else if (diagonalDirections_[directionIndex][0] > 0)
	// base dimension is bound from the left
	queryNearestL1FromSplit<RANGE_LEFT, true, RANGE_LEFT>
	  (vsplit, p, pL1Length, d, directionIndex, minL1Distance,
	   nearestKnownL1Distance, nearest);
      else
	// base dimension is bound from the right
	queryNearestL1FromSplit<RANGE_LEFT, true, RANGE_RIGHT>
	  (vsplit, p, pL1Length, d, directionIndex, minL1Distance,
	   nearestKnownL1Distance, nearest);
    }
    else {
      if (d > 1)
	queryNearestL1FromSplit<RANGE_RIGHT, false, 0>
	  (vsplit, p, pL1Length, d, directionIndex, minL1Distance,
	   nearestKnownL1Distance, nearest);
      else if (diagonalDirections_[directionIndex][0] > 0)
	// base dimension is bound from the left
	queryNearestL1FromSplit<RANGE_RIGHT, true, RANGE_LEFT>
	  (vsplit, p, pL1Length, d, directionIndex, minL1Distance,
	   nearestKnownL1Distance, nearest);
      else
	// base dimension is bound from the right
	queryNearestL1FromSplit<RANGE_RIGHT, true, RANGE_RIGHT>
	  (vsplit, p, pL1Length, d, directionIndex, minL1Distance,
	   nearestKnownL1Distance, nearest);
    }
  }
}

// This is six different cases in one code that is templated for efficiency.
template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
template<int NEAR_SIDE, bool D1, int BASE_BOUND_SIDE /* only used if
							D1 is true */>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
queryNearestL1FromSplit(RangeTreeNode* vsplit, const TPoint& p,
			const TPointElem& pL1Length, int d, int directionIndex,
			const TPointElem& minL1Distance,
			TPointElem& nearestKnownL1Distance, TPoint*& nearest)
{
  const int FAR_SIDE = (NEAR_SIDE + 1) % 2;
  std::list<RangeTreeNode*> edgeNodes;
  std::list<int> subCascadedIndices; // from fractional cascading
  typename std::list<RangeTreeNode*>::iterator iter;
  std::list<int>::iterator cascadedIndexIter;
  RangeTreeNode* leafNode = 0;
  int splitSubIndex;

  // Traverse along the near side of the split, putting the nodes (and
  // indices if D1 is true) in lists to be iterated in reverse order.
  edgeNodes.push_back(vsplit);
  BasicBoundTester<NEAR_SIDE> nearBoundTester(p[d]);
  if (D1) {
   
    splitSubIndex = (BASE_BOUND_SIDE == RANGE_LEFT) ?
	vsplit->lowerLevel_.bls->findFirstGreaterEq(p[0]) :
      vsplit->lowerLevel_.bls->findLastLesserEq(p[0]);
    /*vsplit->lowerLevel_.bls->
      findFirstFromSide<BASE_BOUND_SIDE>(p[0]);*/

    if ((splitSubIndex < 0) ||
	(splitSubIndex >= vsplit->lowerLevel_.bls->getSize()))
      return; // no points in this sub tree are on the searching side of p[0]
    
    CascadeTraverser<NEAR_SIDE, BASE_BOUND_SIDE> traverser(vsplit,
							   splitSubIndex);
    do {
      // push in front to automatically iterate in reverse order 
      edgeNodes.push_front(traverser.getNode());
      subCascadedIndices.push_front(traverser.getCurrentCascadedIndex());
    } while (traverser.goNext(nearBoundTester));
    if (traverser.getNode()->isLeaf())
      leafNode = traverser.getNode();
  }
  else {
    Traverser<NEAR_SIDE> traverser(vsplit, d);
    do {
      // push in front to automatically iterate in reverse order 
      edgeNodes.push_front(traverser.getNode());
    } while (traverser.goNext(nearBoundTester));
    leafNode = traverser.getNode();
  }
  
  // test the leaf node first (it is closest in this dimension)
  if (leafNode != 0 &&
      leafNode->isPointInDirection(p, diagonalDirections_[directionIndex], d)){
    setNearest(nearest, leafNode->point_, directionIndex, pL1Length,
	       nearestKnownL1Distance);
  }
  
  iter = edgeNodes.begin();
  cascadedIndexIter = subCascadedIndices.begin();
  RangeTreeNode* prev = *iter;
  TPointElem offsetP = addInDirection<NEAR_SIDE>(p[d], minL1Distance);
  TPointElem prevMinL1Distance =
    distance<NEAR_SIDE>(offsetP, (*(*iter)->point_)[d]);
  TPointElem currentMinL1Distance;
  
  // iterate starting with the closest in this dimension 
  // (because it has the best chance of having the close points that will
  // narrow the search).
  for (iter++; iter != edgeNodes.end(); iter++) {
    if (prevMinL1Distance >= nearestKnownL1Distance) {
      // no point in going further -- it has already dealt with everything
      // closer than the nearest known distance.
      return;
    }
    
    // add to the minL1Distance for the next level
    currentMinL1Distance =
      distance<NEAR_SIDE>(offsetP, (*(*iter)->point_)[d]);

    if (currentMinL1Distance > nearestKnownL1Distance) {
      // move the split node because the nearestKnownL1Distance was changed
      // to less the current edge node -- so go to the previous edge node.
      vsplit = prev;
      if (vsplit->isLeaf())
	return; // leaf should have been checked already

      if (D1) {
	// reset the splitSubIndex
	splitSubIndex = (BASE_BOUND_SIDE == RANGE_LEFT) ?
	  vsplit->lowerLevel_.bls->findFirstGreaterEq(p[0]) :
	  vsplit->lowerLevel_.bls->findLastLesserEq(p[0]);
	/*vsplit->lowerLevel_.bls->
	  findFirstFromSide<BASE_BOUND_SIDE>(p[0]);*/
	if ((splitSubIndex < 0) ||
	    (splitSubIndex >= vsplit->lowerLevel_.bls->getSize())) {
	  // no points in this sub tree are on the searching side of p[0]
	  return; 
	}
      }
      break;
    }
    if (!D1)
      queryNearestL1(getChild<FAR_SIDE>(prev)->lowerLevel_.rtn, p, pL1Length,
		     d-1, directionIndex, prevMinL1Distance,
		     nearestKnownL1Distance, nearest);
    else if (D1) {
      // base case - find best candidate and test it
      BaseLevelSet* sub = getChild<FAR_SIDE>(prev)->lowerLevel_.bls;

      int cascadedIndex = getCascadedSubIndex<FAR_SIDE, BASE_BOUND_SIDE>
	(prev->lowerLevel_.bls, *cascadedIndexIter, sub);
      // constant time lookup
      TPoint* candidate =
	getExtremePoint<BASE_BOUND_SIDE>(sub, cascadedIndex, directionIndex,
					 numDiagDirections_);
      setNearest(nearest, candidate, directionIndex, pL1Length,
		 nearestKnownL1Distance);
      cascadedIndexIter++;
    }
    
    prevMinL1Distance = currentMinL1Distance;
    prev = *iter;
  }
  
  // now traverse the side furthest from p
  if (!D1) {
    Traverser<FAR_SIDE> traverser(vsplit, d);
    
    while (traverser.goNext
	   (BasicBoundTester<FAR_SIDE>
	    (addInDirection<FAR_SIDE>(offsetP, nearestKnownL1Distance)))) {
      // add to the minL1Distance for the next level
      RangeTreeNode* node = traverser.getNode();
      currentMinL1Distance =
	distance<NEAR_SIDE>(offsetP, (*node->point_)[d]);
      queryNearestL1(getChild<NEAR_SIDE>(node)->lowerLevel_.rtn,
		     p, pL1Length,d-1, directionIndex, currentMinL1Distance,
		     nearestKnownL1Distance, nearest);
    }
    leafNode = traverser.getNode();
  }
  else {
    // base case, d == 1 querying down to d == 0
    CascadeTraverser<FAR_SIDE, BASE_BOUND_SIDE> traverser(vsplit,
							  splitSubIndex);
    
    while (traverser.goNext
	   (BasicBoundTester<FAR_SIDE>
	    (addInDirection<FAR_SIDE>(offsetP, nearestKnownL1Distance)))) {
      RangeTreeNode* node = traverser.getNode();
      BaseLevelSet* sub = getChild<NEAR_SIDE>(node)->lowerLevel_.bls;
      // fractional cascading - constant time lookup
      int index = traverser.getCurrentCascadedIndex();
      index = getCascadedSubIndex<NEAR_SIDE, BASE_BOUND_SIDE>
	(node->lowerLevel_.bls, index, sub);
      TPoint* candidate =
	getExtremePoint<BASE_BOUND_SIDE>(sub, index, directionIndex,
					 numDiagDirections_);
      setNearest(nearest, candidate, directionIndex, pL1Length,
		 nearestKnownL1Distance);
    }
    if (traverser.getNode()->isLeaf())
      leafNode = traverser.getNode();
    else
      leafNode = 0;
  }
  
  // check the far leaf node
  if (leafNode != 0 &&
      leafNode->isPointInDirection(p, diagonalDirections_[directionIndex], d)){
    setNearest(nearest, leafNode->point_, directionIndex, pL1Length,
	       nearestKnownL1Distance);
  }
}

// check the far leaf node
template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
setNearest(TPoint*& nearest, TPoint* candidate, int directionIndex,
	   const TPointElem& pL1Length, TPointElem& nearestKnownL1Distance)
{
  if (candidate == 0)
    return;

  int* direction = diagonalDirections_[directionIndex];
  TPointElem candidateDistance = getL1MetricLength(candidate, direction,
						  DIMENSIONS_) - pL1Length;

  if (candidateDistance < nearestKnownL1Distance) {
    ASSERT(candidateDistance >= 0);
    nearest = candidate;
    nearestKnownL1Distance = candidateDistance;
  }
}

template<class TPoint, class TPointElem, bool ALLOW_NEAREST_NEIGHBOR_QUERY>
void RangeTree<TPoint, TPointElem, ALLOW_NEAREST_NEIGHBOR_QUERY>::
setDiagonalDirections()
{
  numDiagDirections_ = (int)pow(2.0, DIMENSIONS_);
  if (numDiagDirections_ <= 0) return;
  diagonalDirections_ = scinew int*[numDiagDirections_];
  for (int i = 0; i < numDiagDirections_; i++)
    diagonalDirections_[i] = scinew int[DIMENSIONS_];
    
  // set it in the same order as a binary number, but with -1 representing
  // a 0 bit (and 1 representing a 1 bit).
  
  for (int d = 0; d < DIMENSIONS_; d++)
    diagonalDirections_[0][d] = -1;

  int place;
  for (int i = 1; i < numDiagDirections_; i++) {
    place = DIMENSIONS_ - 1;
    while (diagonalDirections_[i-1][place] == 1) {
      diagonalDirections_[i][place] = -1;
      place--;
      ASSERT(place >= 0);
    }
    diagonalDirections_[i][place--] = 1;
    while (place >= 0) {
      diagonalDirections_[i][place] = diagonalDirections_[i-1][place];
      place--;
    }
  }
}

} // End namespace SCIRun

#endif // ndef Core_Containers_RangeTree_h
