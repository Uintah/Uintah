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


#include <Core/Containers/ConsecutiveRangeSet.h>
#include <sstream>
#include <errno.h>

namespace SCIRun {
using namespace std;

const ConsecutiveRangeSet ConsecutiveRangeSet::empty;
const ConsecutiveRangeSet
  ConsecutiveRangeSet::all(INT_MIN, INT_MAX-1 /* -1 to avoid overflow */);  

ConsecutiveRangeSet::ConsecutiveRangeSet(list<int>& set)
  : size_(0) // set to zero just to start
{
  set.sort();
  set.unique();

  if (set.size() == 0) return; // empty set
  
  list<int>::iterator it = set.begin();
  Range range(*it, *it);

  list<Range> rangeSet;
  for (it++; it != set.end(); it++) {
    if ((unsigned long)*it - range.low_ == range.extent_ + 1)
      range.extent_++;
    else {
      // end of range
      rangeSet.push_back(range);
      range = Range(*it, *it);
    }
  }
  rangeSet.push_back(range);

  // this copying just ensures that no more than the necessary
  // size is used
  rangeSet_ = vector<Range>(rangeSet.begin(), rangeSet.end());
  setSize();
}

ConsecutiveRangeSet::ConsecutiveRangeSet(int low, int high)
{
  rangeSet_.reserve(1);
  if (low == INT_MAX || high == INT_MAX)
    cerr << "Warning, initializing ConsectuiveRangeSet with a high of\n"
	 << "INT_MAX may cause overflow.\n";
  rangeSet_.push_back(Range(low, high));
  setSize();
}

// initialize a range set with a string formatted like: "1, 2-8, 10, 15-30"
ConsecutiveRangeSet::ConsecutiveRangeSet(std::string setstr)
  throw(ConsecutiveRangeSetException)
  : size_(0) // set to zero just to start
{
  istringstream in(setstr);
  list<Range> rangeSet;

  int n;
  char c;

  bool isInterval = false;
  int lastNumber = 0;
  bool hasLastNumber = false;
  
  while (!ws(in).eof()) {
    if (in.peek() == '-' && hasLastNumber && !isInterval) {
      in >> c;
      isInterval = true;
    }
    else {
      // try to interpret next part of input as an integer
      in >> n;
    
      if (in.fail()) {
	// not an integer
	in.clear();
	in >> c;
	if (c != ',')
	  throw ConsecutiveRangeSetException(string("ConsecutiveRangeSet cannot parse integer set string: bad character '") + c + "'");
	else {
	  if (isInterval)
	    throw ConsecutiveRangeSetException("ConsecutiveRangeSet cannot parse integer set string: ',' following '-'");
	}
      }
      else if (isInterval) {
	if (!hasLastNumber)
	  throw ConsecutiveRangeSetException("ConsecutiveRangeSet cannot parse integer set string: ambiguous '-'");
	rangeSet.push_back(Range(lastNumber, n));
	isInterval = false;
	hasLastNumber = false;
      }
      else {
	if (hasLastNumber)
	  rangeSet.push_back(Range(lastNumber, lastNumber));
	lastNumber = n;
	hasLastNumber = true;
      }
    }
  }
  if (hasLastNumber)
    rangeSet.push_back(Range(lastNumber, lastNumber));

  rangeSet.sort();

  // check for overlapping intervals
  if (rangeSet.size() == 0) return; // empty set - nothing to do

  list<Range>::iterator it = rangeSet.begin();
  list<Range>::iterator tmp_it;
  int last_high = (*it).high();
  for (it++; it != rangeSet.end(); it++)
  {
    if (last_high + 1 >= (*it).low_) {
      // combine ranges
      int high = (last_high > (*it).high()) ? last_high : (*it).high();
      it--;
      (*it).extent_ = (unsigned long)high - (*it).low_;
      tmp_it = it;
      rangeSet.erase(++tmp_it);
    }
    last_high = (*it).high();
  }

  rangeSet_ = vector<Range>(rangeSet.begin(), rangeSet.end());
  setSize();
}

// Add to the range set, asserting that value is greater or equal
// to anything already in the set.
void ConsecutiveRangeSet::addInOrder(int value)
   throw(ConsecutiveRangeSetException)
{
   if (rangeSet_.size() == 0)
      rangeSet_.push_back(Range(value, value));
   else {
      Range& range = rangeSet_.back();
      int last_value = (int)(range.low_ + range.extent_);
      if (value < last_value) {
	 ostringstream msg;
	 msg << "ConsecutiveRangeSet::addInOrder given value not in order: "
	     << value << " < " << last_value;
	 throw ConsecutiveRangeSetException(msg.str());
      }
      if (value > last_value) {
	 if (value == last_value + 1)
	    range.extent_++;
	 else {
	    // end of range
	    rangeSet_.push_back(Range(value, value));
	 }
      }
      else
	 return; // value == last_value -- don't increment the size
   }
   size_++;
}

void ConsecutiveRangeSet::setSize()
{
  size_ = 0;
  vector<Range>::iterator it = rangeSet_.begin();
  for ( ; it != rangeSet_.end() ; it++) {
    size_ += (*it).extent_ + 1;
  }
}

ConsecutiveRangeSet::iterator& ConsecutiveRangeSet::iterator::operator++()
{
  // check to see if it is already at the end
  CHECKARRAYBOUNDS(range_, 0, (long)set_->rangeSet_.size());
  if (set_->rangeSet_[range_].extent_ > (unsigned long)offset_)
    offset_++;
  else {
    range_++;
    offset_ = 0;
  }
  return *this;
}

bool ConsecutiveRangeSet::operator==(const ConsecutiveRangeSet& set2) const
{
  vector<Range>::const_iterator it = rangeSet_.begin();
  vector<Range>::const_iterator it2 = set2.rangeSet_.begin();
  for (; it != rangeSet_.end() && it2 != set2.rangeSet_.end();
       it++, it2++) {
    if (*it != *it2)
      return false;
  }
  if (it != rangeSet_.end() || it2 != set2.rangeSet_.end())
    return false;
  
  return true;
}

ConsecutiveRangeSet ConsecutiveRangeSet::intersected(const ConsecutiveRangeSet&
						     set2) const
{
  list<Range> newRangeSet;
  Range range(0, 0);
  
  // note that the sets are in sorted order
  vector<Range>::const_iterator it = rangeSet_.begin();
  vector<Range>::const_iterator it2 = set2.rangeSet_.begin();
  while (it != rangeSet_.end() && it2 != set2.rangeSet_.end()) {
    while ((*it).low_ > (*it2).high()) {
      it2++;
      if (it2 == set2.rangeSet_.end())
	return ConsecutiveRangeSet(newRangeSet.begin(), newRangeSet.end());
    }

    while ((*it2).low_ > (*it).high()) {
      it++;
      if (it == rangeSet_.end())
	return ConsecutiveRangeSet(newRangeSet.begin(), newRangeSet.end());
    }

    range.low_ = ((*it).low_ > (*it2).low_) ? (*it).low_ : (*it2).low_;
    int high = ((*it).high() < (*it2).high()) ? (*it).high() : (*it2).high();
    if (high >= range.low_) {
      range.extent_ =  (unsigned long)high - range.low_;
      newRangeSet.push_back(range);
      if ((*it).high() < (*it2).high())
	it++;
      else
	it2++;
    }
  }

  return ConsecutiveRangeSet(newRangeSet.begin(), newRangeSet.end());
}

ConsecutiveRangeSet ConsecutiveRangeSet::unioned(const ConsecutiveRangeSet&
						 set2) const
{
  list<Range> newRangeSet;
  Range range(0, 0);

  //cout << set2.size() << endl;
  if (size() == 0)
    return set2;
  else if (set2.size() == 0)
    return *this;
  
  // note that the sets are in sorted order
  vector<Range>::const_iterator it = rangeSet_.begin();
  vector<Range>::const_iterator it2 = set2.rangeSet_.begin();
  newRangeSet.push_back(((*it).low_ < (*it2).low_) ? *it++ : *it2++);

  while (it != rangeSet_.end() || it2 != set2.rangeSet_.end()) {
    if (it == rangeSet_.end())
      range = *it2++;
    else if (it2 == set2.rangeSet_.end())
      range = *it++;
    else
      range = ((*it).low_ < (*it2).low_) ? *it++ : *it2++;
    
    Range& lastRange = newRangeSet.back();

    // check for overlap
    // being careful about overflow -- for example all == MIN_INT..MAX_INT
    if ((lastRange.high() >= range.low_) ||
	((unsigned long)range.low_ - lastRange.high() == 1)) {
      // combine ranges
      if (range.high() > lastRange.high())
	lastRange.extent_ =  (unsigned long)range.high() - lastRange.low_;
    }
    else
      newRangeSet.push_back(range);
  }

  return ConsecutiveRangeSet(newRangeSet.begin(), newRangeSet.end());
}

string ConsecutiveRangeSet::toString() const
{
  ostringstream stream;
  stream << *this; // << '\0';
  return stream.str();
}

string ConsecutiveRangeSet::expandedString() const
{
   ostringstream stream;
   iterator it = begin();
   if (it != end()) {
      stream << *it;
      for (it++; it != end(); it++)
	 stream << " " << *it;
   }
   return stream.str();
}

ConsecutiveRangeSet::Range::Range(int low, int high)
{
  if (high >= low) {
    low_ = low;
    extent_ = (unsigned long)high - low;
  }
  else {
    low_ = high;
    extent_ = (unsigned long)low - high;
  }
}

ostream& operator<<(ostream& out, const ConsecutiveRangeSet& set)
{
  vector<ConsecutiveRangeSet::Range>::const_iterator it =
    set.rangeSet_.begin();
  
  if (it == set.rangeSet_.end())
    return out; // empty set
  
  (*it).display(out);
  for (++it ; it != set.rangeSet_.end(); it++) {
    out << ", ";
    (*it).display(out);   
  }
  return out;
}
} // End namespace SCIRun

