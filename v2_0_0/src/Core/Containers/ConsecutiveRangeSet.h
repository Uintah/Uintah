/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  ConsecutiveRangeSet.h
 *
 *  Written by:
 *   Wayne Witzel
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 2000
 *
 *  Copyright (C) 2000 SCI Group
 * 
 */

#ifndef SCI_Containers_ConsecutiveRangeSet_h
#define SCI_Containers_ConsecutiveRangeSet_h

#include <sgi_stl_warnings_off.h>
#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <Core/Util/Assert.h>
#include <Core/Exceptions/Exception.h>

namespace SCIRun {

/**************************************

CLASS 
   ConsecutiveRangeSet
   
KEYWORDS
   Interval, integers

DESCRIPTION
   
   Represents a set of integers that are
   stored efficiently if grouped together
   in consecutive ranges.
 
   Written by:
    Wayne Witzel
    Department of Computer Science
    University of Utah
    Nov. 2000
 
   Copyright (C) 2000 SCI Group
  
 
PATTERNS
   
WARNING
  
****************************************/

class ConsecutiveRangeSetException : public Exception
{
public:
  ConsecutiveRangeSetException(std::string msg)
    : msg_(msg) { }

  ConsecutiveRangeSetException(const ConsecutiveRangeSetException& copy)
    : msg_(copy.msg_) { }

  virtual const char* message() const
  { return msg_.c_str(); }
  
  virtual const char* type() const
  { return "ConsecutiveRangeSetException"; }
private:
  std::string msg_;
};

class ConsecutiveRangeSet
{
  friend std::ostream& operator<<(std::ostream& out,
				  const ConsecutiveRangeSet& set);
public:
  
  class iterator
  {
  public:
    iterator(const ConsecutiveRangeSet* set, int range, int offset)
      : set_(set), range_(range), offset_(offset) { }
    iterator(const iterator& it2)
      : set_(it2.set_), range_(it2.range_), offset_(it2.offset_) { }

    iterator& operator=(const iterator& it2) {
      set_ = it2.set_; range_ = it2.range_; offset_ = it2.offset_;
      return *this;
    }
    
    inline int operator*() throw(ConsecutiveRangeSetException);

    bool operator==(const iterator& it2) const
    { return range_ == it2.range_ && offset_ == it2.offset_; }

    bool operator!=(const iterator& it2) const
    { return !(*this == it2); }

    iterator& operator++();
    inline iterator operator++(int);
  private:
    const ConsecutiveRangeSet* set_;
    int range_;
    int offset_;
  };

  
  // represents range: [low, low+extent]
  struct Range
  {
    Range(int low, int high);
    Range(const Range& r2)
      : low_(r2.low_), extent_(r2.extent_) { }

    Range& operator=(const Range& r2)
    { low_ = r2.low_; extent_ = r2.extent_; return *this; }

    bool operator==(const Range& r2) const
    { return low_ == r2.low_ && extent_ == r2.extent_; }

    bool operator!=(const Range& r2) const
    { return low_ != r2.low_ || extent_ != r2.extent_; }
    
    bool operator<(const Range& r2) const
    { return low_ < r2.low_; }

    inline void display(std::ostream& out) const;
	
    int high() const { return (int)(low_ + extent_); }
    int low_;
    unsigned long extent_;
  };

public:
  ConsecutiveRangeSet(std::list<int>& set);

  ConsecutiveRangeSet(int low, int high); // single consecutive range

  ConsecutiveRangeSet() : size_(0) {} // empty set
  
  // initialize a range set with a string formatted like: "1, 2-8, 10, 15-30"
  ConsecutiveRangeSet(std::string setstr) throw(ConsecutiveRangeSetException);

  ConsecutiveRangeSet(const ConsecutiveRangeSet& set2)
    : rangeSet_(set2.rangeSet_), size_(set2.size_) { }
  
  ~ConsecutiveRangeSet() {}

  ConsecutiveRangeSet& operator=(const ConsecutiveRangeSet& set2)
  { rangeSet_ = set2.rangeSet_; size_ = set2.size_; return *this; }

  // Add to the range set, asserting that value is greater or equal
  // to anything already in the set (if it is equal to something already
  // in teh set then the value is simply discarded).
  void addInOrder(int value) throw(ConsecutiveRangeSetException);
  
  template <class AnyIterator>
  void addInOrder(const AnyIterator& begin, const AnyIterator& end)
  { for (AnyIterator it = begin; it != end; ++it) addInOrder(*it); }
  
  bool operator==(const ConsecutiveRangeSet& set2) const;
  bool operator!=(const ConsecutiveRangeSet& set2) const
  { return !(*this == set2); }
      
  // obtain the intersection of two sets
  ConsecutiveRangeSet intersected(const ConsecutiveRangeSet& set2) const;

  // obtain the union of two sets
  ConsecutiveRangeSet unioned(const ConsecutiveRangeSet& set2) const;

  // Could implement binary search on the range set, but this
  // wasn't needed so I didn't do it.  Perhaps in the future if
  // needed. -- Wayne
  //iterator find(int n);
  
  inline iterator begin() const
  { return iterator(this, 0, 0); }

  inline iterator end() const
  { return iterator(this, (int)rangeSet_.size(), 0); }

  unsigned long size() const
  { return size_; }

  std::string toString() const;
 
  // return a space separated list of integers
  std::string expandedString() const;

  // used for debugging
  int getNumRanges()
  { return (int)rangeSet_.size(); }

  static const ConsecutiveRangeSet empty;
  static const ConsecutiveRangeSet all;  
  friend class ConsecutiveRangeSet::iterator;
private:
  template <class InputIterator>
  ConsecutiveRangeSet(InputIterator begin, InputIterator end)
    : rangeSet_(begin, end) { setSize(); }
  void setSize();
  
  std::vector<Range> rangeSet_;
  unsigned long size_; // sum of range (extent+1)'s
};

inline int ConsecutiveRangeSet::iterator::operator*()
  throw(ConsecutiveRangeSetException)
{
  CHECKARRAYBOUNDS(range_, 0, (long)set_->rangeSet_.size());
  return set_->rangeSet_[range_].low_ + offset_;
}

inline
ConsecutiveRangeSet::iterator ConsecutiveRangeSet::iterator::operator++(int)
{
  iterator oldit(*this);
  ++(*this);
  return oldit;
}

inline
void ConsecutiveRangeSet::Range::display(std::ostream& out) const
{
  if (extent_ == 0)
    out << low_;
  else
    out << low_ << " - " << high();
}

} // End namespace SCIRun

#endif
