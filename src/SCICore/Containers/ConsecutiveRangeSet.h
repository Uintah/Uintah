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

#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <SCICore/Util/Assert.h>
#include <SCICore/Exceptions/Exception.h>

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

using namespace SCICore::Exceptions;

class ConsecutiveRangeSetException : public Exception
{
public:
   ConsecutiveRangeSetException(std::string msg)
      : d_msg(msg) { }
   
   ConsecutiveRangeSetException(const ConsecutiveRangeSetException& copy)
      : d_msg(copy.d_msg) { }
   
   virtual const char* message() const
   { return d_msg.c_str(); }
   
   virtual const char* type() const
   { return "SCICore::Containers::ConsecutiveRangeSetException"; }
private:
   std::string d_msg;
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
	 : d_set(set), d_range(range), d_offset(offset) { }
      iterator(const iterator& it2)
	 : d_set(it2.d_set), d_range(it2.d_range), d_offset(it2.d_offset) { }
      
      iterator& operator=(const iterator& it2) {
	 d_set = it2.d_set; d_range = it2.d_range; d_offset = it2.d_offset;
	 return *this;
      }
      
      inline int operator*() throw(ConsecutiveRangeSetException);
      
      bool operator==(const iterator& it2) const
      { return d_range == it2.d_range && d_offset == it2.d_offset; }
      
      bool operator!=(const iterator& it2) const
      { return !(*this == it2); }
      
      iterator& operator++();
      inline iterator operator++(int);
   private:
      const ConsecutiveRangeSet* d_set;
      int d_range;
      int d_offset;
   };

   // represents range: [low, low+extent]
   struct Range
   {
      Range(int low, int high);
      Range(const Range& r2)
	 : d_low(r2.d_low), d_extent(r2.d_extent) { }
      
      Range& operator=(const Range& r2)
      { d_low = r2.d_low; d_extent = r2.d_extent; return *this; }
      
      bool operator==(const Range& r2) const
      { return d_low == r2.d_low && d_extent == r2.d_extent; }

      bool operator!=(const Range& r2) const
      { return d_low != r2.d_low || d_extent != r2.d_extent; }
    
      bool operator<(const Range& r2) const
      { return d_low < r2.d_low; }
      
      inline void display(std::ostream& out) const;
      
      int high() const { return (int)(d_low + d_extent); }
      int d_low;
      unsigned long d_extent;
   };
   
public:
   ConsecutiveRangeSet(std::list<int>& set);
   
   ConsecutiveRangeSet(int low, int high); // single consecutive range
   
   ConsecutiveRangeSet() : d_size(0) {} // empty set
  
   // initialize a range set with a string formatted like: "1, 2-8, 10, 15-30"
   ConsecutiveRangeSet(std::string setstr) throw(ConsecutiveRangeSetException);

   ConsecutiveRangeSet(const ConsecutiveRangeSet& set2)
      : d_rangeSet(set2.d_rangeSet), d_size(set2.d_size) { }
  
   ~ConsecutiveRangeSet() {}

   ConsecutiveRangeSet& operator=(const ConsecutiveRangeSet& set2)
   { d_rangeSet = set2.d_rangeSet; d_size = set2.d_size; return *this; }

   // Add to the range set, asserting that value is greater or equal
   // to anything already in the set.
   void addInOrder(int value) throw(ConsecutiveRangeSetException);
   
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
   { return iterator(this, (int)d_rangeSet.size(), 0); }
   
   unsigned long size() const
   { return d_size; }
   
   std::string toString() const;

   // return a space separated list of integers
   std::string expandedString() const;
   
   // used for debugging
   int getNumRanges()
   { return (int)d_rangeSet.size(); }
   
   static const ConsecutiveRangeSet empty;
   static const ConsecutiveRangeSet all;  
private:
   template <class InputIterator>
   ConsecutiveRangeSet(InputIterator begin, InputIterator end)
      : d_rangeSet(begin, end) { setSize(); }
   void setSize();
   
   std::vector<Range> d_rangeSet;
   unsigned long d_size; // sum of range (extent+1)'s
};

inline int ConsecutiveRangeSet::iterator::operator*()
   throw(ConsecutiveRangeSetException)
{
   CHECKARRAYBOUNDS(d_range, 0, (long)d_set->d_rangeSet.size());
   return d_set->d_rangeSet[d_range].d_low + d_offset;
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
   if (d_extent == 0)
      out << d_low;
   else
      out << d_low << " - " << high();
}

#endif
