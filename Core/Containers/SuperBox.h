#ifndef Core_Containers_BoxGrouper_h
#define Core_Containers_BoxGrouper_h

#define TRY_NEW_WAY

#define SUPERBOX_DEBUGGING
#define SUPERBOX_PERFORMANCE_TESTING

#ifdef SUPERBOX_PERFORMANCE_TESTING
int biggerBoxCount;
int minBiggerBoxCount;
#endif

#include <iostream>

#include <set>
#include <vector>
#include <sci_hash_map.h>
#include <algorithm>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

using namespace std;

// Needed template class properties:
// 
// template class BoxP should provide these methods via "->"
// Point getLow();
// Point getHigh();
// Volume getVolume()
// static Volume getVolume(Point low, Point high);
// Value getArea(int side); -- only needed if using
//                             InternalAreaSuperBoxEvaluator
//
//
// functions for template class Point
// Point Min(Point p1, Point p2); // minimum in each dimension
// Point Max(Point p1, Point p2); // maximum in each dimension
//
// templacte class Evaluator
// Value operator()(set<BoxP>::iterator begin, set<BoxP>::iterator end,
//                  Point low, Point high);
// Note: the value of a set of boxes must not be greater than the value
// of a set of boxes that contains the first set or some internal assumptions
// will fail.
//
// template class Value and Volume
// copy constructor, assignment operator
// construct with 0, assign to 0
//
// template class RangeQuerier
// void query(Point low, Point high, RangeQuerier::ResultContainer& result);
// void queryNeighbors(Point low, Point high, RangeQuerier::ResultContainer& result);
//
// class RangeQuerier::ResultContainer
// follows standard template library container concept
//   (has RangeQuerier::ResultContainer::iterator class, begin(), and end())

using namespace std;

// The default evaluator class.  It evaluates a SuperBox to be the area
// between boxes that are next to each other within the SuperBox.
// Note: To use this, the SuperBox must have an Value getArea(int side)
// function to get the area on the x, y, or z face.
template <class BoxP, class Value>
struct InternalAreaSuperBoxEvaluator
{
  template <class BoxPIterator>
  Value operator()(BoxPIterator beginBoxes, BoxPIterator endBoxes,
		 IntVector low, IntVector high);
};

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
class SuperBox;

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
class BasicBox;

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
class CompositeBox;

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
class SuperBoxSet;

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
class SuperBoxSet
{
public:
  struct BoxHash
  {
    size_t operator()(BoxP box) const
    { return (size_t)box; }
  };
  
  typedef SuperBox<BoxP, Point, Volume, Value, Evaluator> SuperBox;
  typedef BasicBox<BoxP, Point, Volume, Value, Evaluator> BasicBox;
  typedef CompositeBox<BoxP, Point, Volume, Value, Evaluator> CompositeBox;  
  typedef hash_map<BoxP, BasicBox*, BoxHash> BoxHashMap;
  typedef vector<SuperBox*> SuperBoxContainer;
public:
  SuperBoxSet()
  : ownsSuperBoxes_(false), value_(0) {}

  ~SuperBoxSet();
  
  template <class RangeQuerier>
  static SuperBoxSet* makeOptimalSuperBoxSet(vector<BoxP> boxes,
					     RangeQuerier& rangeQuerier)
  { return makeOptimalSuperBoxSet(boxes.begin(), boxes.end(), rangeQuerier); }

  template <class BoxIterator, class RangeQuerier>
  static SuperBoxSet*
  makeOptimalSuperBoxSet(BoxIterator begin, BoxIterator end,
			 RangeQuerier& rangeQuerier)
  { return makeOptimalSuperBoxSet(begin, end, rangeQuerier,
				  false /* no greedy */); }


  // uses greedy approach -- much faster, but may not be optimal
  template <class RangeQuerier>
  static SuperBoxSet* makeNearOptimalSuperBoxSet(vector<BoxP> boxes,
						 RangeQuerier& rangeQuerier)
  { return makeNearOptimalSuperBoxSet(boxes.begin(), boxes.end(),
				  rangeQuerier); }
 
  template <class BoxIterator, class RangeQuerier>
  static SuperBoxSet*
  makeNearOptimalSuperBoxSet(BoxIterator begin, BoxIterator end,
			     RangeQuerier& rangeQuerier)
  { return makeOptimalSuperBoxSet(begin, end, rangeQuerier, true /*greedy*/); }

  
  void addSuperBox(SuperBox* superBox)
  {
    ASSERT(superBox->isValid());
    superBoxes_.push_back(superBox); value_ += superBox->getValue();
  }

  template <class SuperBoxPIterator>
  void addSuperBoxes(SuperBoxPIterator begin, SuperBoxPIterator end)
  {
    for (SuperBoxPIterator iter = begin; iter != end; iter++)
      addSuperBox(*iter);
  }

  const SuperBoxContainer& getSuperBoxes() const
  { return superBoxes_; }

  Value getValue() const
  { return value_; }
  
  void takeOwnershipOfSuperBoxes()
  { ownsSuperBoxes_ = true; }
private:
  template <class BoxIterator, class RangeQuerier>
  static SuperBoxSet*
  makeOptimalSuperBoxSet(BoxIterator begin, BoxIterator end,
			 RangeQuerier& rangeQuerier, bool useGreedyApproach);
  
  SuperBoxSet(const SuperBoxSet&);
  SuperBoxSet& operator=(const SuperBoxSet&);
  
  bool ownsSuperBoxes_;
  SuperBoxContainer superBoxes_;
  Value value_;
};

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
class SuperBox {
public:
  friend class SuperBoxSet<BoxP, Point, Volume, Value, Evaluator>;
  typedef SuperBoxSet<BoxP, Point, Volume, Value, Evaluator> SuperBoxSet;
  typedef BasicBox<BoxP, Point, Volume, Value, Evaluator> BasicBox;
  typedef CompositeBox<BoxP, Point, Volume, Value, Evaluator> CompositeBox;  
  typedef typename SuperBoxSet::BoxHashMap BoxHashMap;

public:

  struct LexCompare {
    bool operator()(const SuperBox* b1, const SuperBox* b2)
    {
      return b1->boxes_ < b2->boxes_;
    } 
  };

  struct ValueCompare {
    bool operator()(const SuperBox* b1, const SuperBox* b2)
    {
      return b1->getValue() == b2->getValue() ? b1 > b2 :
	b1->getValue() > b2->getValue();
    } 
  };

public:
  struct Region {
    Region(Point low, Point high)
      : low_(low), high_(high) {}

    Region& operator=(const Region& copy)
    { low_ = copy.low_; high_ = copy.high_; return *this; }
    
    bool contains(const Region& region) const
    { return (Min(low_, region.low_) == low_ &&
	      Max(high_, region.high_) == high_); }

    bool within(const Region& region) const
    { return region.contains(*this) && !contains(region); }
    
    bool overlaps(const Region& region) const;

    Region enclosingRegion(const Region& region) const
    { return Region(Min(low_, region.low_), Max(high_, region.high_)); }

    Region intersectingRegion(const Region& region) const
    { return Region(Max(low_, region.low_), Min(high_, region.high_)); }

    bool operator==(const Region& region) const
    { return low_ == region.low_ && high_ == region.high_; }
    
    Point low_;
    Point high_;
  };

  template <class SuperBoxPIterator>
  static Region computeEnclosingRegion(SuperBoxPIterator superBoxesBegin,
				       SuperBoxPIterator superBoxesEnd);  
public:
  virtual ~SuperBox()
  {}
  
  const Region& getRegion() const
  { return region_; }

  const Point& getLow() const
  { return region_.low_; }

  const Point& getHigh() const
  { return region_.high_; }

  const Volume& getVolume() const
  { return volume_; }

  const Value& getValue() const
  { return value_; }

  const vector<BoxP>& getBoxes() const
  { return boxes_; }

  virtual vector<BasicBox*> getBasicBoxes(BoxHashMap& boxMap);
  
  // Returns true if this SuperBox conflicts with the given one
  // (if they have an incomplete overlap of Boxes -- some Boxes in
  // common but both have Boxes not in common between them).
  bool conflictsWith(const SuperBox* other) const
  { return conflictsWith(other->getRegion()); }
  
  bool conflictsWith(const Region& region) const;

  // Returns true if this SuperBox contains all of the boxes that
  // the given one contains (in other words, encloses)
  bool contains(const SuperBox* other) const
  { return getRegion().contains(other->getRegion()); }  
  bool contains(Region& region) const
  { return getRegion().contains(region); }
  
  template <class SuperBoxPIterator>
  static Value valueSum(SuperBoxPIterator begin, SuperBoxPIterator end);
  
  template <class RangeQuerier>  
  void getNeighbors(RangeQuerier& rangeQuerier,
		    typename RangeQuerier::ResultContainer& result) const
  { rangeQuerier.queryNeighbors(getLow(), getHigh(), result); }

  virtual void makeActive()
  {} // only composite boxes need to worry about this

  // for debugging
  bool isValid() const;

  template <class RangeQuerier>
  void
  buildActivatedMaximalSuperBoxes(RangeQuerier& rangeQuerier,
				  BoxHashMap& boxMap,
				  vector<SuperBox*>& maximalSuperBoxes,
				  set<CompositeBox*, LexCompare>& allExplored,
				  const Region* withinRegion = 0);  

  template <class BasicBoxPIterator, class RangeQuerier>
  static void
  buildActivatedMaximalSuperBoxes(BasicBoxPIterator begin,
				  BasicBoxPIterator end,
				  RangeQuerier& rangeQuerier,
				  BoxHashMap& boxMap,
				  vector<SuperBox*>& maximalSuperBoxes,
				  const Region* withinRegion = 0);  

  
protected:
  SuperBox(Region region, Volume volume)
    : region_(region), volume_(volume), value_(0)
  {}

  SuperBox(const SuperBox& copy)
    : region_(copy.region_), volume_(copy.volume_), value_(copy.value_),
      boxes_(copy.boxes_)
  {}

  void init(BoxP box)
  { boxes_.push_back(box); init(); }
  
  void init(const vector<BasicBox*>& basicBoxes)
  {
    boxes_.reserve(basicBoxes.size());
    vector<BasicBox*>::const_iterator iter = basicBoxes.begin();
    for (; iter != basicBoxes.end(); ++iter) {
      boxes_.push_back((*iter)->getBox());
    }
    init();
  }
    
  void init();

  // helper for SuperBoxSet::makeOptimalSuperBoxSet()
  // Works basically like a search in a binary tree
  // (pick a SuperBox and consider including it versus excluding it).
  // Unlike a binary search, unfortunately, this search can be somewhat
  // more exhaustive.  However, there is a lot of pruning that saves us.
  template <class RangeQuerier>
  static SuperBoxSet*
  findOptimalSuperBoxSet(RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
			 set<SuperBox*, ValueCompare>& activeBoxes,
			 Value preValue, Value maxPossibleValue,
			 Value currentlyKnownOptimalValue = 0,
			 int depth = 0);

  // quick 'n greedy version
  template <class RangeQuerier>
  static SuperBoxSet*
  findNearOptimalSuperBoxSet(RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
			     set<SuperBox*, ValueCompare>& activeBoxes,
			     Value maxPossibleValue);
  
  // helper for findOptimalSuperBoxSet
  template <class RangeQuerier>
  static void
  takePick(SuperBox* pick, RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
	   set<SuperBox*, ValueCompare>& activeBoxes, Value& maxPossibleValue);
  // helper for findOptimalSuperBoxSet  
  static void undoPick(SuperBox* pick,
		       set<SuperBox*, ValueCompare>& activeBoxes,
		       Value& maxPossibleValue);
  // helper for findOptimalSuperBoxSet  
  static void undoPicks(const vector<SuperBox*>& picks,
			set<SuperBox*, ValueCompare>& activeBoxes,
			Value& maxPossibleValue);
  
  // helper for makeSmallestContainingSuperBox
  template <class RangeQuerier>
  CompositeBox*
  makeSmallestContainingSuperBox(RangeQuerier& rangeQuerier,
				 BoxHashMap& boxMap, BasicBox* neighbor,
				 const Region* withinRegion = 0);

  virtual void makeAvailable() {}
  virtual void makeUnavailable() {}
private:
  Region region_;
  Volume volume_;
  Value value_;
  vector<BoxP> boxes_;
};

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
class BasicBox : public SuperBox<BoxP, Point, Volume, Value, Evaluator> {
  typedef SCIRun::SuperBoxSet<BoxP, Point, Volume, Value, Evaluator> SuperBoxSet;    
  typedef SCIRun::SuperBox<BoxP, Point, Volume, Value, Evaluator> SuperBox;
  typedef typename SuperBox::Region Region;  
  typedef SCIRun::CompositeBox<BoxP, Point, Volume, Value, Evaluator> CompositeBox;
  typedef typename SuperBoxSet::BoxHashMap BoxHashMap;
public:
  BasicBox(BoxP box)
    : SuperBox(Region(box->getLow(), box->getHigh()), box->getVolume()),
      available_(true)
  { init(box); }

  BoxP getBox() const
  { return getBoxes()[0]; }

#if 0
  template <class RangeQuerier>
  void
  buildActivatedMaximalSuperBoxes(RangeQuerier& rangeQuerier,
				  BoxHashMap& boxMap,
				  vector<SuperBox*>& maximalSuperBoxes,
				  const Region* withinRegion = 0);    
#endif
  
  bool isAvailable()
  { return available_; }

  const set<CompositeBox*>& getActiveEnclosingSuperBoxes() const
  { return activeEnclosingSuperBoxes_; }

  // Find any active enclosing SuperBox having the same region as
  // the given region.
  CompositeBox* getActiveEnclosingSuperBox(const Region& region) const;
  
  bool allActiveEnclosingSuperBoxesAlsoEnclose(SuperBox* other) const;

  // If any of the ective enclosing superboxes also enclose other, it
  // will return that active enclosing superbox, otherwise return null.
  CompositeBox* anyActiveEnclosingSuperBoxAlsoEnclosing(SuperBox* other)
    const;
  
  void makeAvailable()
  { available_ = true; }
  
  void makeUnavailable()
  { available_ = false; }

  void addActiveEnclosingSuperBox(CompositeBox* enclosingBox)
  {
    ASSERT(enclosingBox->contains(this));
    activeEnclosingSuperBoxes_.insert(enclosingBox);
  }
  
  void removeActiveEnclosingSuperBox(CompositeBox* inactivatedBox)
  { activeEnclosingSuperBoxes_.erase(inactivatedBox); }

  virtual vector<BasicBox*> getBasicBoxes(BoxHashMap&)
  { return vector<BasicBox*>(1, this); } 
private:
  set<CompositeBox*> activeEnclosingSuperBoxes_;
  bool available_;
};
  
template <class BoxP, class Point, class Volume, class Value, class Evaluator>
class CompositeBox : public SuperBox<BoxP, Point, Volume, Value, Evaluator>  {
  typedef SCIRun::SuperBoxSet<BoxP, Point, Volume, Value, Evaluator> SuperBoxSet;
  typedef SCIRun::SuperBox<BoxP, Point, Volume, Value, Evaluator> SuperBox;    
  typedef typename SuperBox::Region Region;
  typedef typename SuperBox::ValueCompare ValueCompare;
  typedef typename SuperBox::LexCompare LexCompare;    
  typedef SCIRun::BasicBox<BoxP, Point, Volume, Value, Evaluator> BasicBox;
  typedef typename SuperBoxSet::BoxHashMap BoxHashMap;
public:
  CompositeBox(const vector<BasicBox*>& basicBoxes,
	       Region region, Volume totalVolume)
    : SuperBox(region, totalVolume), basicBoxes_(basicBoxes), 
      activeSubSuperBoxMaxValue_(0), parent_(0)
  { init(basicBoxes_); }

  template <class BoxPIterator>
  static CompositeBox* makeCompositeBox(BoxHashMap& boxMap,
					BoxPIterator begin, BoxPIterator end);
  
  ~CompositeBox();

  const set<CompositeBox*, ValueCompare>& getActiveConflicts() const
  { return activeConflicts_; }

  virtual void makeActive();
  
  bool isActive() const
  { return isActive_; }

  void inactivate();

  // inactivate, but create the biggest valid sub-SuperBoxes to
  // make active in its place.
  template <class RangeQuerier>  
  void inactivate(RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
		  set<SuperBox*, ValueCompare>& activeBoxes,
		  Value& maxPossibleValue);
  void reactivate(set<SuperBox*, ValueCompare>& activeBoxes,
		  Value& maxPossibleValue);
  
  // inactivate all conflicts and get back the value lost by doing so
  // (add the values of the conflicts and subtract the values of any
  // sub-SuperBoxes of the conflicts that can become active.
  template <class RangeQuerier>
  void inactivateConflicts(RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
			   set<SuperBox*, ValueCompare>& activeBoxes,
			   Value& maxPossibleValue);

  // reactivate the conflicts.
  void reactivateConflicts(set<SuperBox*, ValueCompare>& activeBoxes,
			   Value& maxPossibleValue);
  
  virtual vector<BasicBox*> getBasicBoxes(BoxHashMap&)
  { return basicBoxes_; }

  // The sub value can never really be more than this one's value.
  // (it only gets calculated as more due to overlapping/conflicting
  // sub SuperBoxes).
  Value getCurrentMaximumPossibleSubValue() const
  { return activeSubSuperBoxMaxValue_ > getValue() ?
      getValue() : activeSubSuperBoxMaxValue_; }
protected:  
  void makeAvailable();
  void makeUnavailable();
  void setParent(CompositeBox* parent)
  { parent_ = parent; }

  static void propogateDeltaMaxValue(CompositeBox* parent, Value delta,
				     Value& maxPossibleValue);  

  void addActiveConflict(CompositeBox* conflict) {
    activeConflicts_.insert(conflict);
    conflict->activeConflicts_.insert(this);
  }
  
  void removeActiveConflict(CompositeBox* inactivatedBox)
  { activeConflicts_.erase(inactivatedBox); }

private:
  void addCreatedSubSuperBoxes(const vector<SuperBox*>& candidates);
  
  bool isActive_;
  vector<BasicBox*> basicBoxes_;
  set<CompositeBox*, ValueCompare> activeConflicts_;

  // sub-SuperBoxes activated when this box was deactivated.
  // When this SuperBox reactivates then these need to deactivate again
  // since this SuperBox encloses them.
  vector<SuperBox*> activatedSubSuperBoxes_;
  Value activeSubSuperBoxMaxValue_;

  // store to be deleted in this CompositeBox's destructor
  vector<SuperBox*> createdSubSuperBoxes_;
  CompositeBox* parent_;
};

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
void SuperBox<BoxP, Point, Volume, Value, Evaluator>::init()
{
  Evaluator evaluator;
  value_ = evaluator(boxes_.begin(), boxes_.end(), getLow(), getHigh());
  ASSERT(isValid());
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
bool SuperBox<BoxP, Point, Volume, Value, Evaluator>::Region::
overlaps(const Region& region) const
{
  Point maxLow = Max(low_, region.low_);
  Point minHigh = Min(high_, region.high_);
  return Min(maxLow, minHigh) == maxLow;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
bool SuperBox<BoxP, Point, Volume, Value, Evaluator>::
conflictsWith(const Region& region) const
{
  return getRegion().overlaps(region) &&
    !getRegion().contains(region) &&
    !region.contains(getRegion());
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class SuperBoxPIterator>
SuperBox<BoxP, Point, Volume, Value, Evaluator>::Region
SuperBox<BoxP, Point, Volume, Value, Evaluator>::
computeEnclosingRegion(SuperBoxPIterator superBoxesBegin,
		       SuperBoxPIterator superBoxesEnd)
{
  ASSERT(superBoxesBegin != superBoxesEnd);

  SuperBoxPIterator iter = superBoxesBegin;
  Region region((*iter)->getLow(), (*iter)->getHigh());
  for (++iter; iter != superBoxesEnd; iter++) {
    region.low_ = Min(region.low_, (*iter)->getLow());
    region.high_ = Max(region.high_, (*iter)->getHigh());
  }
  return region;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
bool SuperBox<BoxP, Point, Volume, Value, Evaluator>::isValid() const
{
  if (boxes_.size() == 0)
    return false; // must contain at least one box

  vector<BoxP>::const_iterator iter = boxes_.begin();
  BoxP box = *iter;
  Point low = box->getLow();
  Point high = box->getHigh();
  Volume totalVolume = box->getVolume();
  for (++iter; iter != boxes_.end(); ++iter)
  {
    box = *iter;
    low = Min(low, box->getLow());
    high = Max(high, box->getHigh());
    totalVolume += box->getVolume();
  }

  // these asserts are just to narrow down the problem
  ASSERT(low == getLow());
  ASSERT(high == getHigh());
  ASSERT(totalVolume == getVolume());
  ASSERT(totalVolume == box->getVolume(low, high) /* static Box function */);

  return low == getLow() && high == getHigh() && totalVolume == getVolume()
    && totalVolume == box->getVolume(low, high) /* static Box function */;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class SuperBoxPIterator>
Value SuperBox<BoxP, Point, Volume, Value, Evaluator>::
valueSum(SuperBoxPIterator begin, SuperBoxPIterator end)
{
  Value result = 0;
  for (SuperBoxPIterator iter = begin; iter != end; iter++) {
    result += (*iter)->getValue();
  }
  return result;
}

// Upper bound of O(2^n * log(n)) which isn't good, but it is more
// likely to be O(n * log(n)) on average, but I don't know how to
// prove that.
template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class RangeQuerier>
SuperBoxSet<BoxP, Point, Volume, Value, Evaluator>*
SuperBox<BoxP, Point, Volume, Value, Evaluator>::
findOptimalSuperBoxSet(RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
		       set<SuperBox*, ValueCompare>& activeBoxes,
		       Value preValue, Value maxPossibleValue,
		       Value currentlyKnownOptimalValue /*= 0*/,
		       int depth /*= 0*/)
{
#ifdef SUPERBOX_DEBUGGING
  Value storeMaxPossibleValue = maxPossibleValue;
  depth++;
  set<SuperBox*, ValueCompare> storeActiveBoxes = activeBoxes;
  typename BoxHashMap::iterator iter = boxMap.begin();
  int numAvailable = 0;
  for ( ; iter != boxMap.end(); iter++) {
    if ((*iter).second->isAvailable())
      numAvailable++;
  }
  cerr << "DEPTH = " << depth << ", numAvailable = " << numAvailable
       << ", maxPossibleValue = " << maxPossibleValue
       << ", currentlyKnownOptimalValue = " << currentlyKnownOptimalValue
       << endl;
  
  /*
  for (set<SuperBox*, ValueCompare>::iterator iter = activeBoxes.begin();
       iter != activeBoxes.end(); iter++) {
    ASSERT(withoutSet.find(*iter) == withoutSet.end());
  }
  */
#endif
  if (activeBoxes.size() == 0)
    return scinew SuperBoxSet();
  
  SuperBox* pick = 0;  
  vector<SuperBox*> keepers;  
  SuperBoxSet* result = 0;
  CompositeBox* compositePick;
  
  // Loop to get all undisputed picks, called keepers.
  // Keepers are the "picked" SuperBoxes that we know are part of
  // the optimal solution at this point (locally optimal).
  do {
    if (pick != 0) {
      // only the last pick is uncertain (the rest are "keepers")
      keepers.push_back(pick);
      preValue += pick->getValue();
    }
    
    if (maxPossibleValue <= currentlyKnownOptimalValue
	&& currentlyKnownOptimalValue > 0 /* make sure there is some other
					     solution in the works */) {
#ifdef SUPERBOX_DEBUGGING
      cerr << depth << "\tPruned\n";
#endif
       
      // Can't beat the current global solution, so don't bother
      // -- great for pruning the search tree.
      undoPicks(keepers, activeBoxes, maxPossibleValue);
      return 0;
    }

    // pick the highest valued SuperBox
    pick = *activeBoxes.begin();

#ifdef SUPERBOX_DEBUGGING
    cerr << depth << "\tPick: " << *pick << endl;
#endif
    
    takePick(pick, rangeQuerier, boxMap, activeBoxes, maxPossibleValue);
    compositePick = dynamic_cast<CompositeBox*>(pick);    
  } while (activeBoxes.size() > 0 &&
	   (compositePick == 0 ||
	    compositePick->getActiveConflicts().size() == 0));

#ifdef SUPERBOX_DEBUGGING
  ASSERT(storeMaxPossibleValue >= maxPossibleValue);
#endif    
  
  // try with pick
  result = findOptimalSuperBoxSet(rangeQuerier, boxMap, activeBoxes,
				  preValue + pick->getValue(),
				  maxPossibleValue, currentlyKnownOptimalValue,
				  depth);
  if (result != 0) {
    result->addSuperBox(pick);
    if (preValue + result->getValue() > currentlyKnownOptimalValue)
      currentlyKnownOptimalValue = preValue + result->getValue();
  }
  
  undoPick(pick, activeBoxes, maxPossibleValue);
  
  // The only reason not to go with the pick is if it conflicts with
  // something that works out better.  So try those.  
  if (compositePick != 0 && compositePick->getActiveConflicts().size() > 0) {
#ifdef SUPERBOX_DEBUGGING
    cerr << depth << "\tWithout: " << *pick << endl;
#endif
    compositePick->inactivate(rangeQuerier, boxMap, activeBoxes,
			      maxPossibleValue);
    
    // Try picking conflicts of the pick starting with the one having the
    // greatest value.
    set<CompositeBox*, ValueCompare>::iterator conflictIter;
    set<CompositeBox*, ValueCompare>::reverse_iterator rConflictIter;
    for (conflictIter = compositePick->getActiveConflicts().begin();
	 conflictIter != compositePick->getActiveConflicts().end();
	 conflictIter++) {
      CompositeBox* conflict = *conflictIter;
#ifdef SUPERBOX_DEBUGGING
      cerr << depth << "\tConflict Pick: " << *conflict << endl;
#endif
      
      // Try each conflict as an alternate possibility to the pick
      takePick(conflict, rangeQuerier, boxMap, activeBoxes, maxPossibleValue);
#ifdef SUPERBOX_DEBUGGING
      ASSERT(storeMaxPossibleValue >= maxPossibleValue);
#endif    
      
      SuperBoxSet* otherPossibility =
	findOptimalSuperBoxSet(rangeQuerier, boxMap, activeBoxes,
			       preValue + conflict->getValue(),
			       maxPossibleValue, currentlyKnownOptimalValue,
			       depth);
      if (otherPossibility != 0)
	otherPossibility->addSuperBox(conflict);
      if (otherPossibility != 0 &&
	  (result == 0 || otherPossibility->getValue() > result->getValue())) {
	delete result;
	result = otherPossibility;
	if (preValue + result->getValue() > currentlyKnownOptimalValue)
	  currentlyKnownOptimalValue = preValue + result->getValue();
      }
      
      undoPick(conflict, activeBoxes, maxPossibleValue);
      
      // inactivate this conflict so efforts won't be repeated when this
      // conflict works with other oonflicts.
      conflict->inactivate(rangeQuerier, boxMap, activeBoxes,
			   maxPossibleValue);
    }
    for (rConflictIter = compositePick->getActiveConflicts().rbegin();
	 rConflictIter != compositePick->getActiveConflicts().rend();
	 rConflictIter++) {
      (*rConflictIter)->reactivate(activeBoxes, maxPossibleValue);
    }
    compositePick->reactivate(activeBoxes, maxPossibleValue);
  }
  undoPicks(keepers, activeBoxes, maxPossibleValue);
  result->addSuperBoxes(keepers.begin(), keepers.end());

#ifdef SUPERBOX_DEBUGGING
  ASSERT(storeMaxPossibleValue == maxPossibleValue);
  ASSERT(storeActiveBoxes == activeBoxes);
#endif
  
  return result;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class RangeQuerier>
SuperBoxSet<BoxP, Point, Volume, Value, Evaluator>*
SuperBox<BoxP, Point, Volume, Value, Evaluator>::
findNearOptimalSuperBoxSet(RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
			   set<SuperBox*, ValueCompare>& activeBoxes,
			   Value maxPossibleValue)
{
#ifdef SUPERBOX_DEBUGGING
  Value storeMaxPossibleValue = maxPossibleValue;
#endif
  
  if (activeBoxes.size() == 0)
    return scinew SuperBoxSet();
  
  SuperBox* pick = 0;  
  vector<SuperBox*> keepers;  
  SuperBoxSet* result = scinew SuperBoxSet();
  
  do {
    // pick the highest valued SuperBox
    pick = *activeBoxes.begin();
    result->addSuperBox(pick);
    takePick(pick, rangeQuerier, boxMap, activeBoxes, maxPossibleValue);
  } while (activeBoxes.size() > 0);

#ifdef SUPERBOX_DEBUGGING
  ASSERT(storeMaxPossibleValue >= maxPossibleValue);
#endif    

  undoPicks(result->getSuperBoxes(), activeBoxes, maxPossibleValue);
  return result;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class RangeQuerier>
void SuperBox<BoxP, Point, Volume, Value, Evaluator>::
takePick(SuperBox* pick, RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
	 set<SuperBox*, ValueCompare>& activeBoxes, Value& maxPossibleValue)
{
  ASSERT(pick != 0);  
  activeBoxes.erase(pick); 
    
  // Make the pick's boxes unavailable for building
  // CompositeBoxes so no conflicts will be introduced.
  pick->makeUnavailable();

  // Deactivate the picks conflicts.  This may introduce sub-SuperBox's
  // of theirs into the active set that don't conflict with anything
  // that's been picked.
  CompositeBox* compositePick = dynamic_cast<CompositeBox*>(pick);
  if (compositePick != 0) {
    compositePick->inactivateConflicts(rangeQuerier, boxMap, activeBoxes,
				       maxPossibleValue);
  }
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
void SuperBox<BoxP, Point, Volume, Value, Evaluator>::
undoPick(SuperBox* pick, set<SuperBox*, ValueCompare>& activeBoxes,
	 Value& maxPossibleValue)
{
  // undo everything that takePick did
  ASSERT(pick != 0);

  CompositeBox* compositePick = dynamic_cast<CompositeBox*>(pick);
  if (compositePick != 0) {
    compositePick->reactivateConflicts(activeBoxes, maxPossibleValue);
  }

  pick->makeAvailable();
  activeBoxes.insert(pick);
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
void SuperBox<BoxP, Point, Volume, Value, Evaluator>::
undoPicks(const vector<SuperBox*>& picks,
	  set<SuperBox*, ValueCompare>& activeBoxes,
	  Value& maxPossibleValue)
{
  vector<SuperBox*>::const_reverse_iterator rIter;
  for (rIter = picks.rbegin(); rIter != picks.rend(); rIter++) {
    undoPick(*rIter, activeBoxes, maxPossibleValue);
  }
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
bool BasicBox<BoxP, Point, Volume, Value, Evaluator>::
allActiveEnclosingSuperBoxesAlsoEnclose(SuperBox* other) const
{
  set<CompositeBox*>::iterator iter;  
  for (iter = getActiveEnclosingSuperBoxes().begin();
       iter != getActiveEnclosingSuperBoxes().end(); iter++) {
    if (!(*iter)->contains(other))
      return false;
  }
  return true;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
CompositeBox<BoxP, Point, Volume, Value, Evaluator>*
BasicBox<BoxP, Point, Volume, Value, Evaluator>::
anyActiveEnclosingSuperBoxAlsoEnclosing(SuperBox* other) const
{
  set<CompositeBox*>::iterator iter;  
  for (iter = getActiveEnclosingSuperBoxes().begin();
       iter != getActiveEnclosingSuperBoxes().end(); iter++) {
    if ((*iter)->contains(other))
      return (*iter);
  }
  return 0;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
CompositeBox<BoxP, Point, Volume, Value, Evaluator>*
BasicBox<BoxP, Point, Volume, Value, Evaluator>::
getActiveEnclosingSuperBox(const Region& region) const
{
  set<CompositeBox*>::iterator iter;  
  for (iter = getActiveEnclosingSuperBoxes().begin();
       iter != getActiveEnclosingSuperBoxes().end(); iter++) {
    if ((*iter)->getRegion() == region)
      return (*iter);
  }
  return 0;

}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class BoxPIterator>
CompositeBox<BoxP, Point, Volume, Value, Evaluator>*
CompositeBox<BoxP, Point, Volume, Value, Evaluator>::
makeCompositeBox(BoxHashMap& boxMap, BoxPIterator begin, BoxPIterator end)
{
  ASSERT(begin != end);
  BoxPIterator iter = begin;
  BoxP box = *iter;
  Point low = box->getLow();
  Point high = box->getHigh();
  vector<BasicBox*> basicBoxes;  
  BasicBox* basicBox = boxMap[box];
  basicBoxes.push_back(basicBox);
  ASSERT(basicBox->isAvailable());
  Volume totalVolume = (*iter)->getVolume();
  for (++iter; iter != end; iter++) {
    BoxP box = *iter;
    BasicBox* basicBox = boxMap[box];
    basicBoxes.push_back(basicBox);
    ASSERT(basicBox->isAvailable());
    low = Min(low, box->getLow());
    high = Max(high, box->getHigh());
    totalVolume += box->getVolume();
  }
  return scinew CompositeBox(basicBoxes, Region(low, high), totalVolume);
}


template <class BoxP, class Point, class Volume, class Value, class Evaluator>
CompositeBox<BoxP, Point, Volume, Value, Evaluator>::~CompositeBox()
{
  // should have reactivated before deleting
  ASSERT(activatedSubSuperBoxes_.size() == 0);

  if (isActive_)
    inactivate();

  for (vector<SuperBox*>::iterator iter = createdSubSuperBoxes_.begin();
       iter != createdSubSuperBoxes_.end(); iter++)
    delete *iter;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
void CompositeBox<BoxP, Point, Volume, Value, Evaluator>::
inactivate()
{
  // shouldn't inactivate twice
  ASSERT(activatedSubSuperBoxes_.size() == 0);
  isActive_ = false;

  set<CompositeBox*, ValueCompare>::iterator conflictIter;
  for (conflictIter = activeConflicts_.begin();
       conflictIter != activeConflicts_.end(); conflictIter++) {
    (*conflictIter)->activeConflicts_.erase(this);
  }

  vector<BasicBox*>::iterator iter;
  for (iter = basicBoxes_.begin(); iter != basicBoxes_.end(); iter++) {
    (*iter)->removeActiveEnclosingSuperBox(this);
  }
}


template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class RangeQuerier>
void CompositeBox<BoxP, Point, Volume, Value, Evaluator>::
inactivate(RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
	   set<SuperBox*, ValueCompare>& activeBoxes,
	   Value& maxPossibleValue)
{
  ASSERT(activeSubSuperBoxMaxValue_ == 0);
  inactivate();
  propogateDeltaMaxValue(parent_, -getValue(), maxPossibleValue);
  
  SuperBox::buildActivatedMaximalSuperBoxes(basicBoxes_.begin(),
					    basicBoxes_.end(), rangeQuerier,
					    boxMap, activatedSubSuperBoxes_,
					    &getRegion());
#if 0
  vector<BasicBox*>::iterator iter;    
  for (iter = basicBoxes_.begin(); iter != basicBoxes_.end(); iter++) {
    if ((*iter)->isAvailable()) {
      (*iter)->buildActivatedMaximalSuperBoxes(rangeQuerier, boxMap,
					       activatedSubSuperBoxes_,
					       &getRegion());
    }
  }
#endif
  
  activeBoxes.erase(this);  
  activeBoxes.insert(activatedSubSuperBoxes_.begin(),
		     activatedSubSuperBoxes_.end());
  for (vector<SuperBox*>::iterator subIter = activatedSubSuperBoxes_.begin();
       subIter != activatedSubSuperBoxes_.end(); subIter++) {
    CompositeBox* compositeBox = dynamic_cast<CompositeBox*>(*subIter);
    if (compositeBox != 0)
      compositeBox->setParent(this);
    propogateDeltaMaxValue(this, (*subIter)->getValue(), maxPossibleValue);
  }
  addCreatedSubSuperBoxes(activatedSubSuperBoxes_);
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
void CompositeBox<BoxP, Point, Volume, Value, Evaluator>::
reactivate(set<SuperBox*, ValueCompare>& activeBoxes,
	   Value& maxPossibleValue)
{
  isActive_ = true;

  vector<SuperBox*>::iterator subIter;
  for (subIter = activatedSubSuperBoxes_.begin();
       subIter != activatedSubSuperBoxes_.end(); subIter++) {
    activeBoxes.erase(*subIter);
    CompositeBox* compositeBox = dynamic_cast<CompositeBox*>(*subIter);
    if (compositeBox != 0)
      compositeBox->inactivate();
    propogateDeltaMaxValue(this, -(*subIter)->getValue(), maxPossibleValue);
  }
  activeBoxes.insert(this);
  propogateDeltaMaxValue(parent_, getValue(), maxPossibleValue);

  activatedSubSuperBoxes_.clear();  
  
  vector<BasicBox*>::iterator iter;
  for (iter = basicBoxes_.begin(); iter != basicBoxes_.end(); iter++) {
    (*iter)->addActiveEnclosingSuperBox(this);
  }

  set<CompositeBox*, ValueCompare>::iterator conflictIter;
  for (conflictIter = activeConflicts_.begin();
       conflictIter != activeConflicts_.end(); conflictIter++) {
    ASSERT(conflictsWith(*conflictIter));
    (*conflictIter)->activeConflicts_.insert(this);
  }
  
  ASSERT(activeSubSuperBoxMaxValue_ == 0);  
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
void CompositeBox<BoxP, Point, Volume, Value, Evaluator>::
makeUnavailable()
{
  isActive_ = false;
  set<CompositeBox*, ValueCompare>::iterator conflictIter;
  for (conflictIter = activeConflicts_.begin();
       conflictIter != activeConflicts_.end(); conflictIter++) {
    (*conflictIter)->activeConflicts_.erase(this);
  }

  vector<BasicBox*>::iterator iter;
  for (iter = basicBoxes_.begin(); iter != basicBoxes_.end(); iter++) {
    (*iter)->makeUnavailable();
  }
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
void CompositeBox<BoxP, Point, Volume, Value, Evaluator>::
makeAvailable()
{
  isActive_ = true;
  set<CompositeBox*, ValueCompare>::iterator conflictIter;
  for (conflictIter = activeConflicts_.begin();
       conflictIter != activeConflicts_.end(); conflictIter++) {
    ASSERT(conflictsWith(*conflictIter));
    (*conflictIter)->activeConflicts_.insert(this);
  }
  
  vector<BasicBox*>::iterator iter;
  for (iter = basicBoxes_.begin(); iter != basicBoxes_.end(); iter++) {
    (*iter)->makeAvailable();
  }
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
void CompositeBox<BoxP, Point, Volume, Value, Evaluator>::
addCreatedSubSuperBoxes(const vector<SuperBox*>& candidates)
{
  createdSubSuperBoxes_.reserve(createdSubSuperBoxes_.size() +
				candidates.size());
  for (vector<SuperBox*>::const_iterator iter = candidates.begin();
       iter != candidates.end(); iter++) {
    // could only have created CompositeBoxes, not BasicBoxes
    if (dynamic_cast<CompositeBox*>(*iter) != 0)
      createdSubSuperBoxes_.push_back(*iter);
  }

}

#if 0
template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class RangeQuerier>
void BasicBox<BoxP, Point, Volume, Value, Evaluator>::
buildActivatedMaximalSuperBoxes(RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
				vector<SuperBox*>& maximalSuperBoxes,
				const Region* withinRegion /* = 0 */)
{
#ifdef SUPERBOX_PERFORMANCE_TESTING
  minBiggerBoxCount++;
#endif
  
  set<CompositeBox*, LexCompare> allExploredBoxes;
  
  SuperBox::buildActivatedMaximalSuperBoxes(rangeQuerier, boxMap, maximalSuperBoxes,
					    allExploredBoxes, withinRegion);
  
  // delete temporary ones
  set<CompositeBox*, LexCompare>::iterator iter;
  for (iter = allExploredBoxes.begin(); iter != allExploredBoxes.end(); iter++)
  {
    if (!(*iter)->isActive())
      delete *iter;
  }
}
#endif

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class BasicBoxPIterator, class RangeQuerier>
void SuperBox<BoxP, Point, Volume, Value, Evaluator>::
buildActivatedMaximalSuperBoxes(BasicBoxPIterator begin,
				BasicBoxPIterator end,
				RangeQuerier& rangeQuerier,
				BoxHashMap& boxMap,
				vector<SuperBox*>& maximalSuperBoxes,
				const Region* withinRegion = 0)
{
  set<CompositeBox*, LexCompare> allExploredBoxes;
  for (BasicBoxPIterator iter = begin; iter != end; iter++) {
    if ((*iter)->isAvailable()) {
#ifdef SUPERBOX_PERFORMANCE_TESTING
      minBiggerBoxCount++;
#endif
      (*iter)->buildActivatedMaximalSuperBoxes(rangeQuerier, boxMap,
					       maximalSuperBoxes,
					       allExploredBoxes, withinRegion);
    }
  }
  
  // delete temporary ones
  set<CompositeBox*, LexCompare>::iterator iter;
  for (iter = allExploredBoxes.begin(); iter != allExploredBoxes.end(); iter++)
  {
    if (!(*iter)->isActive())
      delete *iter;
  }
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class RangeQuerier>
void SuperBox<BoxP, Point, Volume, Value, Evaluator>::
buildActivatedMaximalSuperBoxes(RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
				vector<SuperBox*>& maximalSuperBoxes,
				set<CompositeBox*, LexCompare>& allExplored,
				const Region* withinRegion /* = 0 */)
{
  typedef typename RangeQuerier::ResultContainer RangeContainer;
#ifdef SUPERBOX_PERFORMANCE_TESTING
  biggerBoxCount++;
#endif

  bool hasBiggerBox = false;
  
#ifdef TRY_NEW_WAY
  ASSERT(getBasicBoxes(boxMap).size() >= 1);
  BasicBox* anyBasicBox = getBasicBoxes(boxMap)[0];
  CompositeBox* anyActiveEnclosingSuperBox =
    anyBasicBox->anyActiveEnclosingSuperBoxAlsoEnclosing(this);

  RangeContainer neighbors;
  if (anyActiveEnclosingSuperBox != 0) {
    anyActiveEnclosingSuperBox->getNeighbors(rangeQuerier, neighbors);
    hasBiggerBox = true;
  }
  else {
    getNeighbors(rangeQuerier, neighbors);
  }

  unsigned int i;
  vector< pair<BasicBox*, Region> > minimalNeighbors;
  typename RangeContainer::iterator it = neighbors.begin();
  for (; it != neighbors.end(); it++) {
    BasicBox* neighbor = boxMap[*it];
    if (neighbor == 0) {
      // allows for queriers that cn return results outside of the
      // group of boxes we are concerned about (they are just ignored)
      continue;
    }
    if (!neighbor->isAvailable())
      continue;
    Region newRegion = getRegion().enclosingRegion(neighbor->getRegion());
    for (i = 0; i < minimalNeighbors.size(); i++) {
      if (newRegion.contains(minimalNeighbors[i].second))
	break;
      else if (minimalNeighbors[i].second.contains(newRegion)) {
	// replace with smaller region
	minimalNeighbors[i] = make_pair(neighbor, newRegion);
	break;
      }
    }
    if (i == minimalNeighbors.size()) {
      // add a new one
      minimalNeighbors.push_back(make_pair(neighbor, newRegion));
    }
  }
  for (i = 0; i < minimalNeighbors.size(); i++) {
    BasicBox* neighbor = minimalNeighbors[i].first;
  
#else
  RangeContainer neighbors;
  getNeighbors(rangeQuerier, neighbors);

  typename RangeContainer::iterator it = neighbors.begin();
  for (; it != neighbors.end(); it++) {
    BasicBox* neighbor = boxMap[*it];

    // check to see if joining with this neighbor brings hope of new
    // possibilities.
    if (neighbor->getActiveEnclosingSuperBoxes().size() >= 1 &&
	neighbor->allActiveEnclosingSuperBoxesAlsoEnclose(this)) {
      // Joining with the neighbor doesn't create any new possibilites
      // for maximal SuperBoxes that won't be discovered in other ways.
      // This is derived from the fact, which I proved to myself, that
      // for SuperBox m to be a maximalSuperBox, it must contain two
      // neighboring boxes such that neither are contained in any other
      // maximal SuperBox or one of them is contained in a maximalSuperBox
      // that the other one is not contained in.  If no such neighboring
      // boxes existed, then all of the boxes in m would be contained
      // in some other SuperBox, so m wouldn't be maximal.
      hasBiggerBox = true;
      continue;
    }
#endif
    
    CompositeBox* newSuperBox =
      makeSmallestContainingSuperBox(rangeQuerier, boxMap, neighbor,
				     withinRegion);    
    if (newSuperBox != 0) {
      hasBiggerBox = true;      
      if (neighbor->getActiveEnclosingSuperBox(newSuperBox->getRegion()) != 0){
	// newSuperBox is nothing new
	delete newSuperBox;
      }
      else if (allExplored.insert(newSuperBox).second == false) {
	// possibility has already been explored
	delete newSuperBox;
      }
      else {
	newSuperBox->buildActivatedMaximalSuperBoxes(rangeQuerier, boxMap,
						     maximalSuperBoxes,
						     allExplored,
						     withinRegion);
      }
    }
  }

  if (!hasBiggerBox) {
    // It still might have a bigger box outside of withinRegion,
    // so check for that first.
    if (withinRegion != 0) {
      BasicBox* box = boxMap[boxes_.front()];
      if (box->anyActiveEnclosingSuperBoxAlsoEnclosing(this))
	return; // has bigger box
    }

    maximalSuperBoxes.push_back(this);
    makeActive();
#ifdef SUPERBOX_DEBUGGING
    cerr << "Active SuperBox: " << *this << endl;
#endif
  }
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class RangeQuerier>
CompositeBox<BoxP, Point, Volume, Value, Evaluator>*
SuperBox<BoxP, Point, Volume, Value, Evaluator>::
makeSmallestContainingSuperBox(RangeQuerier& rangeQuerier,
			       BoxHashMap& boxMap, BasicBox* neighbor,
			       const Region* withinRegion /* = 0 */)
{   
  typedef typename RangeQuerier::ResultContainer RangeContainer;
  vector<BasicBox*> basicBoxes;
  if (!neighbor->isAvailable())
    return 0;  // a box is unavailable, so superbox can't be made

  Point low = Min(getLow(), neighbor->getLow());
  Point high = Max(getHigh(), neighbor->getHigh());
  if (withinRegion != 0 && !Region(low, high).within(*withinRegion))
    return 0; // can't be larger than maximum region
  
  Volume totalVolume = getVolume() + neighbor->getVolume();
  BoxP dummyBox = 0;
  Volume enclosedVolume = dummyBox->getVolume(low, high) /* static Box 
							    function */;

  if (totalVolume == enclosedVolume) {
    basicBoxes = getBasicBoxes(boxMap);
    basicBoxes.push_back(neighbor);
    return scinew CompositeBox(basicBoxes, Region(low, high), totalVolume);
  }
  
  unsigned long prevNumBoxes = 2;
  do {
    if (withinRegion != 0 && !Region(low, high).within(*withinRegion))
      return 0; // can't be larger than maximum region
    
    RangeContainer boxes;
    rangeQuerier.query(low, high, boxes);
    if (prevNumBoxes >= (unsigned long)boxes.size())
      return 0; // nothing new found -- no enclosing super box exists
    prevNumBoxes = boxes.size();
    basicBoxes.clear();
    basicBoxes.reserve(boxes.size());
    
    totalVolume = 0;
    typename RangeContainer::iterator iter = boxes.begin();
    for (; iter != boxes.end(); iter++) {
      BoxP box = *iter;
      BasicBox* basicBox = boxMap[box];
      basicBoxes.push_back(basicBox);
      if (!basicBox->isAvailable())
	return 0; // a box is unavailable, so superbox can't be made

      low = Min(low, box->getLow());
      high = Max(high, box->getHigh());
      totalVolume += box->getVolume();
    }
    BoxP dummyBox = 0;
    enclosedVolume = dummyBox->getVolume(low, high) /* static Box function */;
  } while (totalVolume != enclosedVolume);

  // totalVolume == enclosedVolume so a new superbox was found
  return scinew CompositeBox(basicBoxes, Region(low, high), totalVolume);
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
void CompositeBox<BoxP, Point, Volume, Value, Evaluator>::
makeActive()
{
  isActive_ = true;
  for (vector<BasicBox*>::iterator iter = basicBoxes_.begin();
       iter != basicBoxes_.end(); ++iter) {
    const set<CompositeBox*>& conflicts =
      (*iter)->getActiveEnclosingSuperBoxes();
    set<CompositeBox*>::const_iterator conflictIter = conflicts.begin();
    for ( ; conflictIter != conflicts.end(); ++conflictIter) {
      CompositeBox* conflict = *conflictIter;
      ASSERT(conflictsWith(conflict));
      addActiveConflict(conflict);
    }
    
    (*iter)->addActiveEnclosingSuperBox(this);
  }
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class RangeQuerier>  
void CompositeBox<BoxP, Point, Volume, Value, Evaluator>::
inactivateConflicts(RangeQuerier& rangeQuerier, BoxHashMap& boxMap,
		    set<SuperBox*, ValueCompare>& activeBoxes,
		    Value& maxPossibleValue)
{
  set<CompositeBox*>::iterator iter;
  for (iter = activeConflicts_.begin(); iter != activeConflicts_.end();
       iter++) {
#ifdef SUPERBOX_DEBUGGING
    cerr << "\tConflict: " << **iter << endl;
#endif
    (*iter)->inactivate(rangeQuerier, boxMap,
			activeBoxes, maxPossibleValue);
  }
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
void CompositeBox<BoxP, Point, Volume, Value, Evaluator>::
reactivateConflicts(set<SuperBox*, ValueCompare>& activeBoxes,
		    Value& maxPossibleValue)
{
  set<CompositeBox*>::reverse_iterator rIter;  
  for (rIter = activeConflicts_.rbegin(); rIter != activeConflicts_.rend();
       rIter++) {
    (*rIter)->reactivate(activeBoxes, maxPossibleValue);
  }
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
void CompositeBox<BoxP, Point, Volume, Value, Evaluator>::
propogateDeltaMaxValue(CompositeBox* parent, Value delta,
		       Value& maxPossibleValue)
{
  CompositeBox* ancestor = parent;
  while (ancestor != 0 && delta != 0) {
    Value prevMaxPossibleSubValue =
      ancestor->getCurrentMaximumPossibleSubValue();
    ancestor->activeSubSuperBoxMaxValue_ += delta;
    delta =
      ancestor->getCurrentMaximumPossibleSubValue() - prevMaxPossibleSubValue;
    ancestor = ancestor->parent_;
  } 
  maxPossibleValue += delta;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
template <class BoxIterator, class RangeQuerier>
SuperBoxSet<BoxP, Point, Volume, Value, Evaluator>*
SuperBoxSet<BoxP, Point, Volume, Value, Evaluator>::
makeOptimalSuperBoxSet(BoxIterator begin, BoxIterator end,
		       RangeQuerier& rangeQuerier, bool useGreedyApproach)
{
  int n = 0;
  for (BoxIterator it = begin; it != end; it++)
    n++; // count number of boxes

  vector<BasicBox*>::iterator b_iter;
  vector<SuperBox*>::iterator sb_iter;

  vector<BasicBox*> basicBoxes(n);
  BoxHashMap boxMap(n);
 
  int i = 0;
  for (BoxIterator it = begin; it != end; it++, i++) {
    BoxP box = *it;
    BasicBox* basicBox = scinew BasicBox(box);
    boxMap[box] = basicBox;
    basicBoxes[i] = basicBox;
  }

  vector<SuperBox*> maximalSuperBoxes;
#if 0
  for (unsigned long i = 0; i < basicBoxes.size(); i++) {
    cerr << i << endl;
    basicBoxes[i]->buildActivatedMaximalSuperBoxes(rangeQuerier, boxMap, 
						   maximalSuperBoxes);
  }
#endif
  SuperBox::buildActivatedMaximalSuperBoxes(basicBoxes.begin(),
					    basicBoxes.end(), rangeQuerier,
					    boxMap, maximalSuperBoxes);

#ifdef SUPERBOX_DEBUGGING
  cerr << "Maximal SuperBoxes:\n";
  for (sb_iter = maximalSuperBoxes.begin(); sb_iter != maximalSuperBoxes.end();
       sb_iter++) {
    cerr << "\t" << **sb_iter << endl;
  }
#endif

  Value totalPossibleValue =
    SuperBox::valueSum(maximalSuperBoxes.begin(), maximalSuperBoxes.end());

  set<SuperBox*, SuperBox::ValueCompare> activeBoxes(maximalSuperBoxes.begin(),
						     maximalSuperBoxes.end());
  SuperBoxSet* superBoxSet;
  if (useGreedyApproach) {
    superBoxSet = SuperBox::findNearOptimalSuperBoxSet(rangeQuerier, boxMap,
						       activeBoxes,
						       totalPossibleValue);

  }
  else {
    superBoxSet = SuperBox::findOptimalSuperBoxSet(rangeQuerier, boxMap,
						   activeBoxes, 0,
						   totalPossibleValue);
  }
  
  // Copy the other SuperBoxes to a new SuperBoxSet.  These new SuperBoxes
  // will be simple SuperBoxes without the extraneous garbage of BasicBox
  // and CompositeBox.
  SuperBoxSet* result = scinew SuperBoxSet();
  SuperBoxContainer::const_iterator const_iter;
  for (const_iter = superBoxSet->getSuperBoxes().begin();
       const_iter != superBoxSet->getSuperBoxes().end(); const_iter++) {
    result->addSuperBox(scinew SuperBox(**const_iter));
  }
  
  // Take ownership of these super boxes so they will be deleted with the
  // result is deleted.
  result->takeOwnershipOfSuperBoxes();

  // delete the temporary SuperBoxes that were created
  for (b_iter = basicBoxes.begin(); b_iter != basicBoxes.end(); b_iter++)
    delete *b_iter;

  
  for (sb_iter = maximalSuperBoxes.begin(); sb_iter != maximalSuperBoxes.end();
       sb_iter++) {
    // Only delete CompositeBox maximalSuperBoxes since the basicBoxes
    // were already deleted.
    delete dynamic_cast<CompositeBox*>(*sb_iter);
  }

  delete superBoxSet;
  return result;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
SuperBoxSet<BoxP, Point, Volume, Value, Evaluator>::~SuperBoxSet()
{
  if (ownsSuperBoxes_) {
    for (unsigned int i = 0; i < superBoxes_.size(); i++)
      delete superBoxes_[i];
  }
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
vector<BasicBox<BoxP, Point, Volume, Value, Evaluator>*>
SuperBox<BoxP, Point, Volume, Value, Evaluator>::
getBasicBoxes(BoxHashMap& boxMap)
{
  // only really implemented for completeness
  vector<BasicBox*> result;
  result.reserve(boxes_.size());
    
  for (vector<BoxP>::iterator iter = boxes_.begin(); iter != boxes_.end();
       iter++) {
    result.push_back(boxMap[*iter]);
  }
  return result;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
ostream&
operator<<(ostream& out,
	   const SuperBox<BoxP, Point, Volume, Value, Evaluator>& superBox)
{
  ASSERT(superBox.isValid());

  out << superBox.getBoxes().size() << ": ";
  Point low = superBox.getLow();
  Point high = superBox.getHigh();
  ::operator<<(out, low);
  out << " - ";
  ::operator<<(out, high);
  /*
  if (superBox.getBoxes().size() == 0) {
    out << "()";
    return out;
  }

  out << "(";
  vector<BoxP>::const_iterator iter = superBox.getBoxes().begin();
  out << (*iter)->getID();
  
  for (++iter; iter != superBox.getBoxes().end(); ++iter) {
    out << ", " << (*iter)->getID();
  }
  out << ")";
  */
  
  return out;
}

template <class BoxP, class Point, class Volume, class Value, class Evaluator>
ostream&
operator<<(ostream& out, const
	   SuperBoxSet<BoxP, Point, Volume, Value, Evaluator>& superBoxSet)
{
  typedef SuperBoxSet<BoxP, Point, Volume, Value, Evaluator> SuperBoxSet;
  typedef SuperBoxSet::SuperBoxContainer SuperBoxContainer;

  SuperBoxContainer::const_iterator iter;
  for (iter = superBoxSet.getSuperBoxes().begin();
       iter != superBoxSet.getSuperBoxes().end(); iter++) {
    out << *(*iter) << endl;
  }
  return out;
}

template <class BoxP, class Value>
template <class BoxPIterator>
Value InternalAreaSuperBoxEvaluator<BoxP, Value>::
operator()(BoxPIterator beginBoxes, BoxPIterator endBoxes,
	   IntVector low, IntVector /*high*/)
{
  Value value = 0;
  for (BoxPIterator iter = beginBoxes; iter != endBoxes; iter++) {
    // Add each of the 3 low sides that are not on the edge.
    
    // Only the low side is included because if both low and high
    // sides were considered than the area would just be counted twice
    // (each bit of internal area has a box on the low side and a box
    // on the high side).
    BoxP box = *iter;
    for (int i = 0; i < 3; i++) {
      if (box->getLow()[i] > low[i]) {
	// low i-side is internal, so include its area.
	
	// get the area by multiplying the length of the two other edges
	Value area = box->getArea(i);
	value += area;
      }
    }
  }

  return value;
}

} // End namespace SCIRun

#endif // ndef Core_Containers_BoxGrouper_h

