#include "ConsecutiveRangeSet.h"
#include <sstream>
#include <errno.h>

using namespace std;

const ConsecutiveRangeSet ConsecutiveRangeSet::empty;
const ConsecutiveRangeSet
  ConsecutiveRangeSet::all(INT_MIN, INT_MAX-1 /* -1 to avoid overflow */);  

ConsecutiveRangeSet::ConsecutiveRangeSet(list<int>& set)
  : d_size(0) // set to zero just to start
{
  set.sort();
  set.unique();

  if (set.size() == 0) return; // empty set
  
  list<int>::iterator it = set.begin();
  Range range(*it, *it);

  list<Range> rangeSet;
  for (it++; it != set.end(); it++) {
    if ((unsigned long)*it == range.d_low + range.d_extent + 1)
      range.d_extent++;
    else {
      // end of range
      rangeSet.push_back(range);
      range = Range(*it, *it);
    }
  }
  rangeSet.push_back(range);

  // this copying just ensures that no more than the necessary
  // size is used
  d_rangeSet = vector<Range>(rangeSet.begin(), rangeSet.end());
  setSize();
}

ConsecutiveRangeSet::ConsecutiveRangeSet(int low, int high)
{
  d_rangeSet.reserve(1);
  if (low == INT_MAX || high == INT_MAX)
    cerr << "Warning, initializing ConsectuiveRangeSet with a high of\n"
	 << "INT_MAX may cause overflow.\n";
  d_rangeSet.push_back(Range(low, high));
  setSize();
}

// initialize a range set with a string formatted like: "1, 2-8, 10, 15-30"
ConsecutiveRangeSet::ConsecutiveRangeSet(std::string setstr)
  throw(ConsecutiveRangeSetException)
  : d_size(0) // set to zero just to start
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
    if ((unsigned long)last_high + 1 >= (unsigned long)(*it).d_low) {
      // combine ranges
      int high = (last_high > (*it).high()) ? last_high : (*it).high();
      it--;
      (*it).d_extent = (unsigned long)high - (*it).d_low;
      tmp_it = it;
      rangeSet.erase(++tmp_it);
    }
    last_high = (*it).high();
  }

  d_rangeSet = vector<Range>(rangeSet.begin(), rangeSet.end());
  setSize();
}

void ConsecutiveRangeSet::setSize()
{
  d_size = 0;
  vector<Range>::iterator it = d_rangeSet.begin();
  for ( ; it != d_rangeSet.end() ; it++) {
    d_size += (unsigned long)(*it).d_extent + 1;
  }
}

ConsecutiveRangeSet::iterator& ConsecutiveRangeSet::iterator::operator++()
{
  // check to see if it is already at the end
  CHECKARRAYBOUNDS(d_range, 0, (long)d_set->d_rangeSet.size());
  if (d_set->d_rangeSet[d_range].d_extent > d_offset)
    d_offset++;
  else {
    d_range++;
    d_offset = 0;
  }
  return *this;
}

bool ConsecutiveRangeSet::operator==(const ConsecutiveRangeSet& set2) const
{
  vector<Range>::const_iterator it = d_rangeSet.begin();
  vector<Range>::const_iterator it2 = set2.d_rangeSet.begin();
  for (; it != d_rangeSet.end() && it2 != set2.d_rangeSet.end(); it++, it2++) {
    if (*it != *it2)
      return false;
  }
  
  return true;
}

ConsecutiveRangeSet ConsecutiveRangeSet::intersected(const ConsecutiveRangeSet&
						     set2) const
{
  list<Range> newRangeSet;
  Range range(0, 0);
  
  // note that the sets are in sorted order
  vector<Range>::const_iterator it = d_rangeSet.begin();
  vector<Range>::const_iterator it2 = set2.d_rangeSet.begin();
  while (it != d_rangeSet.end() && it2 != set2.d_rangeSet.end()) {
    while ((*it).d_low > (*it2).high()) {
      it2++;
      if (it2 == set2.d_rangeSet.end())
	return ConsecutiveRangeSet(newRangeSet.begin(), newRangeSet.end());
    }

    while ((*it2).d_low > (*it).high()) {
      it++;
      if (it == d_rangeSet.end())
	return ConsecutiveRangeSet(newRangeSet.begin(), newRangeSet.end());
    }

    range.d_low = ((*it).d_low > (*it2).d_low) ? (*it).d_low : (*it2).d_low;
    int high = ((*it).high() < (*it2).high()) ? (*it).high() : (*it2).high();
    if (high >= range.d_low) {
      range.d_extent =  (unsigned long)high - range.d_low;
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
  vector<Range>::const_iterator it = d_rangeSet.begin();
  vector<Range>::const_iterator it2 = set2.d_rangeSet.begin();
  newRangeSet.push_back(((*it).d_low < (*it2).d_low) ? *it++ : *it2++);

  while (it != d_rangeSet.end() || it2 != set2.d_rangeSet.end()) {
    if (it == d_rangeSet.end())
      range = *it2++;
    else if (it2 == set2.d_rangeSet.end())
      range = *it++;
    else
      range = ((*it).d_low < (*it2).d_low) ? *it++ : *it2++;
    
    Range& lastRange = newRangeSet.back();

    // check for overlap
    if ((unsigned long)lastRange.high() + 1 >= (unsigned long)range.d_low) {
      // combine ranges
      if (range.high() > lastRange.high())
	lastRange.d_extent =  (unsigned long)range.high() - lastRange.d_low;
    }
    else
      newRangeSet.push_back(range);
  }

  return ConsecutiveRangeSet(newRangeSet.begin(), newRangeSet.end());
}

string ConsecutiveRangeSet::toString() const
{
  ostringstream stream;
  stream << *this << '\0';
  return stream.str();
}

ConsecutiveRangeSet::Range::Range(int low, int high)
{
  if (high >= low) {
    d_low = low;
    d_extent = (unsigned long)high - low;
  }
  else {
    d_low = high;
    d_extent = (unsigned long)low - high;
  }
}

ostream& operator<<(ostream& out, const ConsecutiveRangeSet& set)
{
  vector<ConsecutiveRangeSet::Range>::const_iterator it =
    set.d_rangeSet.begin();
  
  if (it == set.d_rangeSet.end())
    return out; // empty set
  
  (*it).display(out);
  for (++it ; it != set.d_rangeSet.end(); it++) {
    out << ", ";
    (*it).display(out);   
  }
  return out;
}
