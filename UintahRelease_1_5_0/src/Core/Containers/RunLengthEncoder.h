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

#ifndef SCI_Containers_RunLengthEncoder_h
#define SCI_Containers_RunLengthEncoder_h

#include <vector>
#include <list>
#include <iosfwd>
#include <sstream>
#include <string>

#ifndef _WIN32
#include <unistd.h>
#else
typedef long ssize_t;
#endif

#include <cerrno>

#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Endian.h>
#include <Core/Util/SizeTypeConvert.h>

namespace SCIRun {

  //  template <class T> class list;
  
/**************************************

CLASS
   RunLengthEncoder
   
   Encodes and decodes lists of type T given a Sequencer class which
   finds sequenced "runs" in the list.  It can also write and read
   the encoded data to/from a file, or seek a specific index in such
   a file.

   There are two Sequencer classes defined in this file:
   EqualElementSequencer and EqualIntervalSequencer.  There is also
   a DefaultRunLengthSequencer which simply inherits from
   EqualElementSequencer but is specialized for some primitive
   data types to use the EqualIntervalSequencer instead.  You can
   make other specialization of the DefaultRunLengthSequencer for
   other types (that's what it is there for).

   The RunLengthEncoder expects certain things from any Sequencer.
   First, Sequencer::SequenceRule must be a type containing whatever
   information may be needed for it to know how a sequence goes.
   Sequencer must have the following static methods:

   static bool needRule();
     Indicates whether a SequenceRule is needed (if not then the
     RunLengthEncoder won't need to store it).

   static T getSequenceItem(T start, SequenceRule rule, int i);
     Gives the ith element of a sequence given the first item and
     the rule for that sequence.

   And it must specify the following member methods:

   void addItem(T item, bool canUseFromPreviousRun);
   const RunSpec<T, SequenceRule>& getCurrentRun();
     These two methods go together.  It should work as follows.
     The RunLengthEncoder adds items to the Sequencer.  The Sequencer
     must keep track of the current run.  As items are added, the
     current run will either grow in length if the sequence continues
     or it will start a new run.  If it must start a new run, the new
     run may contain part of the old run only if canUseFromPreviousRun
     is true, otherwise it must be a run of length 1. getCurrentRun()
     should return whatever run includes the last item that was added
     (even if the run length is only 1 and therefore not an official
     run yet).  The RunLengthEncoder will actually only treat
     these as official runs if their length is greater than some
     minimum run length which is determined by the sizes of T and
     SequenceRule (the minimum run length is the number of elements
     a run must have in order for it to be more efficiently stored
     as a run in a separate group).  canUseFromPreviousRun will be
     true iff the previous run was not at the minimum run length yet.

   Note: You should make sure that T has an appropriately defined copy
   constructor and assignment operator as well as any operators needed
   by the Sequencer.
   
GENERAL INFORMATION

   RunLengthEncoder.h

   Wayne Witzel
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   RLE, RunLength, Encoding, Compression

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  
template<class T>
class DefaultRunLengthSequencer;

template <class T, class SequenceRule>
struct RunSpec
{
  RunSpec()
    : length_(0) {}
  
  T start_;
  SequenceRule rule_;
  unsigned long length_;
};
  
template<class T, class Sequencer = DefaultRunLengthSequencer<T>
                        /* see basic Sequencer classes below */>
class RunLengthEncoder
{
public:
  typedef typename Sequencer::SequenceRule SequenceRule;

private:  
  struct Group
  {
    Group() // default empty group
      : data_(1), length_(0) {}
    
    // make a Run
    Group(const RunSpec<T, SequenceRule>& run)
      : data_(1, run.start_), sequenceRule_(run.rule_),
	length_(run.length_) 
    { }

    // make a non-run group from the first n items of the list
    // popping items off the list as it goes.
    Group(typename std::list<T>& itemList, unsigned long n)
      : data_(n), sequenceRule_(), length_(n)
    {
      for (unsigned long i = 0; i < n; i++) {
	data_[i] = itemList.front();
	itemList.pop_front();
      }
    }
    
    // runs have a single data item indicating the first item, with
    // squenceRule_ specifying how the sequence goes from there.
    bool isRun() const 
    { return data_.size() == 1 && length_ > 1; }

    std::vector<T> data_;

    // If the group is a sequence then then data_[0] and rule_ are
    // used to find the element of the sequence for any index
    SequenceRule sequenceRule_;

    unsigned long length_; /* # of items this group represents
			       (redundant for non-sequences) */
  };

public:
  class iterator;        // For C++ (AIX xlC only?) nested classes are 
  friend class iterator; // private, thus we need to make them a friend.

  class iterator
  {
  private:
    typedef typename std::list<Group>::const_iterator GroupListIterator;
  public:
    iterator(GroupListIterator groupIter, unsigned long groupIndex = 0)
      : groupIter_(groupIter), groupIndex_(groupIndex) {}
    iterator(const iterator& iter)
      : groupIter_(iter.groupIter_), groupIndex_(iter.groupIndex_) {}
    ~iterator() {}

    iterator& operator=(const iterator& iter)
    { groupIter_ = iter.groupIter_; groupIndex_ = iter.groupIndex_;
      return *this; }
    
    bool operator==(const iterator& it2) const
    { return groupIter_ == it2.groupIter_ &&
	groupIndex_ == it2.groupIndex_; }

    bool operator!=(const iterator& it2) const
    { return !(*this == it2); }
    
    inline T operator*();

    inline iterator& operator++();
    inline iterator& operator--();

    iterator operator++(int)
    {
      iterator oldit(*this);
      ++(*this);
      return oldit;
    }

    iterator operator--(int)
    {
      iterator oldit(*this);
      --(*this);
      return oldit;
    }
  private:
    GroupListIterator groupIter_;
    unsigned long groupIndex_;
  };

public:
  RunLengthEncoder()
    : size_(0)
  { }

  template<class InputIterator>
  RunLengthEncoder(InputIterator begin, InputIterator end)
    : size_(0)
  { addItems(begin, end); }
   
  RunLengthEncoder(std::istream& in, bool swapBytes = false /* endianness swap */,
		   int nByteMode = sizeof(unsigned long)  /* 32bit/64bit
							     conversion */)
    : size_(0)
  { read(in, swapBytes, nByteMode); }

  void addItem(T item);

  template<class InputIterator>
  void addItems(InputIterator begin, InputIterator end)
  {
    for (InputIterator it = begin; it != end; it++)
      addItem(*it);
  }

  // Copy data out to another container, the amount that the other
  // container can hold or the amount contained in the RunLengthEncoder
  // (whichever comes first).
  template<class InputIterator>
  void copyOut(InputIterator it_begin, InputIterator it_end)
  {
    iterator rle_iter = begin();
    for (InputIterator it = it_begin; (it != it_end) && (rle_iter != end());
	 it++, rle_iter++)
      *it = *rle_iter;
  }

  // You will not be able to iterate through some of the last
  // data items added until this finalize method is called
  // (write and testPrint will automatically call finalize,
  // and you don't need to call finalize for read or seek).
  void finalize();

  bool isFinalized() const
  { return considerationItems_.size() == 0; }

  iterator begin() const
  { return iterator(groups_.begin(), 0); }

  iterator end() const throw(InternalError)
  {
    if (!isFinalized())
      throw InternalError("You cannot iterate through RunLengthEncoder items until the RunLengthEncoder has been finalized.", __FILE__, __LINE__);
    return iterator(groups_.end(), 0);
  }

  unsigned long size()
  { return size_; }

  // these return the number of bytes written/read
  long write(std::ostream& out) throw(ErrnoException);

  long read(std::istream& in, bool swapBytes = false,
	    int nByteMode = sizeof(unsigned long)) throw(InternalError)
  {
    if (swapBytes || nByteMode != sizeof(unsigned long))
      return readPriv<true>(in, swapBytes, nByteMode);
    else
      return readPriv<false>(in, swapBytes, nByteMode);
  }

  // seek for and read a single item from a file
  static T seek(int fd /* file descriptor */, unsigned long index,
		bool swapBytes = false, int nByteMode = sizeof(unsigned long))
    throw(InternalError)
  {
    if (swapBytes || nByteMode != sizeof(unsigned long))
      return seekPriv<true>(fd, index, swapBytes, nByteMode);
    else
      return seekPriv<false>(fd, index, swapBytes, nByteMode);
  }
 
  void testPrint( std::ostream & out );

private:

  // should optimize itself when no conversion is needed
  template <bool needConversion>
  long readPriv(std::istream& in, bool swapBytes, int nByteMode)
    throw(InternalError);

  template <bool needConversion>
  static T seekPriv(int fd /* file descriptor */, unsigned long index,
		    bool swapBytes, int nByteMode)
    throw(InternalError);

  template<class SIZE_T>
  static inline void readSizeType(std::istream& in, bool needConversion, bool swapBytes,
			   int nByteMode, SIZE_T& s);

  template<class SIZE_T>
  static inline void readSizeType(int fd, bool needConversion, bool swapBytes,
			   int nByteMode, SIZE_T& s);

  template<class SIZE_T>
  inline void pReadSizeType(int fd, bool needConversion, bool swapBytes,
			    int nByteMode, off_t offset, SIZE_T& s);

  void write(int fd, void* data, ssize_t size) throw(ErrnoException)
  {
    if (::write(fd, data, size) != size)
      throw ErrnoException("RunLengthEncoder::write", errno, __FILE__, __LINE__);
  }

  inline static ssize_t pread(int fd, void* buf, size_t nbyte, off_t offset)
  {
    return pread(fd, buf, nbyte, offset);
  }

  inline static off_t lseek(int fd, off_t offset, int whence)
  {
    return ::lseek(fd, offset, whence);
  }

  inline static unsigned long getMinRunLength(bool isDefaultRuleRun)
  { return (header_item_size + sizeof(T) + ruleStorageSize(isDefaultRuleRun)) /
      sizeof(T) + 1; /* should compile out to a constant, I hope */}

  inline static ssize_t ruleStorageSize(bool isDefaultRuleRun)
  { return (ssize_t)(Sequencer::needRule() && !isDefaultRuleRun ?
		     sizeof(SequenceRule) : 0); }

  // Temporary place to put items until it figures out what kind
  // of group it belongs in.
  std::list<T> considerationItems_;

  // This is where the data is permanently stored and grouped.
  std::list<Group> groups_;
  unsigned long size_; // total number of items added

  Sequencer sequencer_;
  
  static ssize_t header_item_size;
};


/* Basic sequencers to use with a RunLengthEncoders */

// EqualElementSequencer for compressing runs of equal elements only.
template <class T>
class EqualElementSequencer
{
public:
  // these don't really matter since no rule is needed
  typedef short SequenceRule;
  static short defaultSequenceRule;
public:
  EqualElementSequencer()
  { }
    
  void addItem(T item, bool canUseFromPreviousRun);
  const RunSpec<T, SequenceRule>& getCurrentRun()
  { return currentRun_; }

  inline static T getSequenceItem(T start, SequenceRule /*rule*/,
				  int /*index*/)
  { return start; }

  inline static bool needRule()
  { return false; }

  inline static bool isEqual(T t1, T t2);
private:
  RunSpec<T, SequenceRule> currentRun_;
};

template <class T>
short EqualElementSequencer<T>::defaultSequenceRule = 0; // arbitrary


// EqualElementSequencer for compressing equal interval runs.
// The following operations must be defined for this to work:
// Tdiff(0) constructor
// Tdiff operator-(T, T)
// Tdiff operator*(Tdiff, int)
// T operator+(T, Tdiff)
template <class T, class Tdiff = T>
class EqualIntervalSequencer
{
public:
  typedef Tdiff SequenceRule;
  static Tdiff defaultSequenceRule;
public:
  void addItem(T item, bool canUseFromPreviousRun);
  const RunSpec<T, SequenceRule>& getCurrentRun()
  { return currentRun_; }
  
  inline static T getSequenceItem(T start, SequenceRule rule,
				  int index)
  { return start + rule * index; }
  
  inline static bool needRule()
  { return true; }
private:
  RunSpec<T, SequenceRule> currentRun_;
  T lastItem_;
};

template <class T, class Tdiff>
Tdiff EqualIntervalSequencer<T, Tdiff>::defaultSequenceRule(0);

/* Default Sequencer with specializations for basic types */
template <class T>
class DefaultRunLengthSequencer : public EqualElementSequencer<T>
{ };

template<>
class DefaultRunLengthSequencer<int> : public EqualIntervalSequencer<int>
{ };

template<>
class DefaultRunLengthSequencer<long> : public EqualIntervalSequencer<long>
{ };

template<>
class DefaultRunLengthSequencer<unsigned long>
  : public EqualIntervalSequencer<unsigned long>
{ };

template<>
class DefaultRunLengthSequencer<float> : public EqualIntervalSequencer<float>
{ };

template<>
class DefaultRunLengthSequencer<double> : public EqualIntervalSequencer<double>
{ };


template<class T, class Sequencer>
ssize_t RunLengthEncoder<T, Sequencer>::header_item_size =
  sizeof(unsigned long) + sizeof(ssize_t);

/* Equal Element Sequencer */
template <class T>
void EqualElementSequencer<T>::addItem(T item, bool /*canUseFromPreviousRun*/)
{
  if (currentRun_.length_ > 0) {
    if (isEqual(item, currentRun_.start_)) {
      currentRun_.length_++;
      return;
    }
  }

  currentRun_.start_ = item;
  currentRun_.length_ = 1;
}

template <class T>
inline bool EqualElementSequencer<T>::isEqual(T t1, T t2)
{
  // default is for a byte-wise comparison
  int n = sizeof(T);
  char* byte1 = (char*)&t1;
  char* byte2 = (char*)&t2;
  while (n > 0) {
    if (*byte1 != *byte2)
      return false;
    byte1++;
    byte2++;
    n--;
  }
  return true;
}

/* Specialize isEqual for the primitives */

template<>
inline bool EqualElementSequencer<int>::isEqual(int t1, int t2)
{ return t1 == t2; }

template<>
inline bool EqualElementSequencer<long>::isEqual(long t1, long t2)
{ return t1 == t2; }

template<>
inline bool EqualElementSequencer<unsigned long>::isEqual(unsigned long t1,
							  unsigned long t2)
{ return t1 == t2; }

template<>
inline bool EqualElementSequencer<float>::isEqual(float t1, float t2)
{ return t1 == t2; }

template<>
inline bool EqualElementSequencer<double>::isEqual(double t1, double t2)
{ return t1 == t2; }

template<>
inline bool EqualElementSequencer<char>::isEqual(char t1, char t2)
{ return t1 == t2; }

/* Equal Interval Sequencer */
template <class T, class Tdiff>
void EqualIntervalSequencer<T, Tdiff>::addItem(T item,
					       bool canUseFromPreviousRun)
{
  if (currentRun_.length_ > 1) {
    // the sequence pattern has started
    if (getSequenceItem(currentRun_.start_, currentRun_.rule_,
			(int)currentRun_.length_) != item) {
      // sequence broken - start a new one
      if (canUseFromPreviousRun) {
	currentRun_.start_ = lastItem_;
	currentRun_.rule_ = item - lastItem_;
	if (getSequenceItem(currentRun_.start_, currentRun_.rule_, 1)
	    != item) {
	  // something like a floating point round-off error -- don't do run
	  currentRun_.start_ = item;
	  currentRun_.length_ = 0; // will become 1
	}
	else
	  currentRun_.length_ = 1; // will become 2
      }
      else {
	currentRun_.start_ = item;
	currentRun_.length_ = 0; // will become 1
      }
    } 
  }
  else if (currentRun_.length_ > 0) {
    currentRun_.rule_ = item - currentRun_.start_;
    if (getSequenceItem(currentRun_.start_, currentRun_.rule_, 1) != item){
      // something like a floating point round-off error -- don't do run 
      currentRun_.start_ = item;
      currentRun_.length_ = 0; // will become 1, 2 otherwise
    } 
  }
  else {
    currentRun_.start_ = item;
  }
  currentRun_.length_++;
  lastItem_ = item;
}


/* RunLengthEncoder methods */

template<class T, class Sequencer>
void RunLengthEncoder<T, Sequencer>::addItem(T item)
{
  sequencer_.addItem(item, considerationItems_.size() > 0);
  const RunSpec<T, SequenceRule>& currentRun = sequencer_.getCurrentRun();
  bool isDefaultRule = (currentRun.rule_ == Sequencer::defaultSequenceRule);
  if (currentRun.length_ == getMinRunLength(isDefaultRule)) {
    // start a new run (long enough to be officially called a run)

    if (considerationItems_.size() + 1 > currentRun.length_) {
      // flush non-sequence garbage (or at least, sequences not long enough
      // to make it) that are beyond consideration.
      unsigned long doneConsiderationLength =
	considerationItems_.size() + 1 - currentRun.length_;
      groups_.push_back(Group(considerationItems_, doneConsiderationLength));
    } 

    ASSERT(considerationItems_.size() + 1 == currentRun.length_);
    considerationItems_.clear();
    groups_.push_back(Group(currentRun));
  }
  else if (currentRun.length_ > getMinRunLength(isDefaultRule)) {
    // current run is growing
    groups_.back().length_++;
  }      
  else {
    // the current run is not long enough yet, keep it in the
    // considerations queue.
    considerationItems_.push_back(item);    
  }

  size_++;
}

template<class T, class Sequencer>
void RunLengthEncoder<T, Sequencer>::finalize()
{
  if (considerationItems_.size() > 0) {
    groups_.push_back(Group(considerationItems_,
			     considerationItems_.size()));
    ASSERT(considerationItems_.size() == 0);
  }
}


/***************************************************
 * RunLengthEncoder file format (in binary format):
 *
 * <header_item>*
 * [end_data_position][end_index == size_]
 * <data_item>*
 *
 * where each group has a header_item and data_item in sequential order
 * defined as follows:
 *
 * <header_item> := [start_data_pos][start_index]
 * <data_item> := [start_data][data_incr] |  // for runs
 *          [data1][data2]...[data(length)][possible_padding] // for non-runs
 *
 * Data positions are specified relative to the top of the header (the
 * location of the file where it first starts writing).  The start data
 * position for the group is found by taking the end_data_pos of the
 * previous group (or the top [first_data_pos] for the first group).
 *
 * When reading, it figures out if a group is a run by checking to see
 * if length * sizeof(T) != end_data_pos - start_data_pos
 * (where end_data_pos is the start of the next one).  This works by
 * the fact that it only makes runs if it will save space (so the amount
 * of data storage will be different than for storing a non-run of the
 * same length).
 *
 */

template<class T, class Sequencer>
long RunLengthEncoder<T, Sequencer>::write(std::ostream& out) throw(ErrnoException)
{
  finalize();
  ssize_t header_size = (ssize_t)(groups_.size() + 1) * header_item_size;
  ssize_t data_pos = header_size;
  unsigned long index = 0;

  // write the header  
  for (typename std::list<Group>::iterator groupIter = groups_.begin();
       groupIter != groups_.end(); groupIter++) {
    // write each header item (one per group):
    // [start_data_pos][start_index]
    Group& group = *groupIter;
    out.write((char*)&data_pos, sizeof(ssize_t));
    out.write((char*)&index, sizeof(unsigned long));
    index += group.length_;     
    data_pos += sizeof(T) * group.data_.size();
    if (Sequencer::needRule() && group.isRun()) {
      bool isDefaultRuleRun =
	(group.sequenceRule_ == Sequencer::defaultSequenceRule);
      data_pos += ruleStorageSize(isDefaultRuleRun);
    }
  }

  ASSERT(index == size_);
  // write final part of header:
  // [end_data_pos][end_index]
  out.write((char*)&data_pos, sizeof(ssize_t));
  out.write((char*)&index, sizeof(unsigned long));
  
  // write the data
  for (typename std::list<Group>::iterator groupIter = groups_.begin();
       groupIter != groups_.end(); groupIter++) {
    Group& group = *groupIter;
    out.write((char*)&group.data_[0],
	  (ssize_t)(sizeof(T) * group.data_.size()));

    // for runs, it writes a single starting data item followed by the
    // increment
    if (Sequencer::needRule() && group.isRun() &&
	(group.sequenceRule_ != Sequencer::defaultSequenceRule))
      out.write((char*)&group.sequenceRule_, ruleStorageSize(false));
  }
  
  return data_pos; // returns the number of bytes written
}


template<class T, class Sequencer>
template<class SIZE_T> // should either be unsigned long or ssize_t
inline void RunLengthEncoder<T, Sequencer>::
readSizeType(std::istream& in, bool needConversion, bool swapBytes, int nByteMode,
	     SIZE_T& s)
{
  if (needConversion) {
    uint64_t s64;
    in.read((char*)&s64, nByteMode);
    s = (SIZE_T)convertSizeType(&s64, swapBytes, nByteMode);
  }
  else {
    in.read((char*)&s, nByteMode);
  }
}

template<class T, class Sequencer>
template<class SIZE_T> // should either be unsigned long or ssize_t
inline void RunLengthEncoder<T, Sequencer>::
readSizeType(int fd, bool needConversion, bool swapBytes, int nByteMode,
	     SIZE_T& s)
{
  if (needConversion) {
    uint64_t s64;
    ::read(fd, (char*)&s64, nByteMode);
    s = (SIZE_T)convertSizeType(&s64, swapBytes, nByteMode);
  }
  else {
    ::read(fd, (char*)&s, nByteMode);
  }
}

template<class T, class Sequencer>
template<class SIZE_T> // should either be unsigned long or ssize_t
inline void RunLengthEncoder<T, Sequencer>::
pReadSizeType(int fd, bool needConversion, bool swapBytes, int nByteMode,
	      off_t offset, SIZE_T& s)
{
  if (needConversion) {
    uint64_t s64;
    pread(fd, (char*)&s64, nByteMode, offset);
    s = (SIZE_T)convertSizeType(&s64, swapBytes, nByteMode);
  }
  else {
    pread(fd, (char*)&s, nByteMode, offset);
  }
}

template<class T, class Sequencer>
template<bool needConversion>
long RunLengthEncoder<T, Sequencer>::readPriv(std::istream& in, bool swapBytes,
					      int nByteMode)
  throw(InternalError)
{
  // assume the header item's are composed of unsigned longs and ssize_t's
  ASSERT((RunLengthEncoder<T, Sequencer>::header_item_size %
	  sizeof(unsigned long)) == 0);
  unsigned long header_item_size = nByteMode *
    RunLengthEncoder<T, Sequencer>::header_item_size / sizeof(unsigned long);
  
  considerationItems_.clear();
  groups_.clear();
   
  ssize_t header_size;
  ssize_t start_data_pos;
  ssize_t end_data_pos = 0;
  unsigned long start_index;
  unsigned long end_index=0;
  
  readSizeType(in, needConversion, swapBytes, nByteMode, start_data_pos);
  readSizeType(in, needConversion, swapBytes, nByteMode, start_index);
  header_size = start_data_pos;

  if (header_size % header_item_size != 0)
    throw InternalError("Invalid RunLengthEncoded data", __FILE__, __LINE__);
   
  unsigned long num_runs_left = header_size / header_item_size - 1;
  std::vector<bool> usesDefaultRule(num_runs_left, false);

  // read the header
  int i = 0;
  for (; num_runs_left > 0; num_runs_left--, i++) {
    groups_.push_back(Group());
    Group& group = groups_.back();

    readSizeType(in, needConversion, swapBytes, nByteMode, end_data_pos);
    readSizeType(in, needConversion, swapBytes, nByteMode, end_index);

    group.length_ = end_index - start_index;
    
    if (group.length_ * sizeof(T) == 
	(unsigned long)(end_data_pos - start_data_pos))
      // not a run -- signify by resizing the data
      group.data_.resize((end_data_pos - start_data_pos) / sizeof(T));
    else if (Sequencer::needRule() &&
	     (end_data_pos - start_data_pos == sizeof(T))) {
      // must be default sequence rule
      usesDefaultRule[i] = true;
    }

    start_data_pos = end_data_pos;
    start_index = end_index;
  }

  size_ = end_index;

  // read the data
  i = 0;
  for (typename std::list<Group>::iterator groupIter = groups_.begin();
       groupIter != groups_.end(); groupIter++, i++) {      
    std::vector<T>& data = (*groupIter).data_;
    if ((*groupIter).isRun()) {
      in.read((char*)&data[0], sizeof(T));
      if (needConversion && swapBytes) SCIRun::swapbytes(data[0]);
      if (Sequencer::needRule()) {
	if (usesDefaultRule[i])
	  (*groupIter).sequenceRule_ = Sequencer::defaultSequenceRule;
	else {
	  in.read((char*)&(*groupIter).sequenceRule_, ruleStorageSize(false));
	  if (needConversion && swapBytes)
	    SCIRun::swapbytes((*groupIter).sequenceRule_);
	}
      }
    }
    else
    {
      ASSERT(data.size() == (*groupIter).length_);
      in.read((char*)&data[0], (long)(sizeof(T) * data.size()));
      if (needConversion && swapBytes) {
	for (unsigned long index = 0; index < data.size(); index++) {
	  SCIRun::swapbytes(data[index]);
	}
      }
    }
  }

  return end_data_pos;
}

template<class T, class Sequencer>
template<bool needConversion>
T RunLengthEncoder<T, Sequencer>::seekPriv(int fd, unsigned long index,
					   bool swapBytes, int nByteMode)
  throw(InternalError)
{
  ssize_t header_item_size = RunLengthEncoder<T, Sequencer>::header_item_size
    / sizeof(unsigned long) * nByteMode;
  
  // does a binary type search in the header
  ssize_t start = lseek(fd, 0, SEEK_CUR);

  // the header is as follows:
  // [start_data_pos][index]
  // [start_data_pos][index]
  // ...
  // [end_data_pos][# of elements]
    
  ssize_t header_size;
  readSizeType(fd, needConversion, swapBytes, nByteMode, header_size);

  if (header_size % header_item_size != 0)
    throw InternalError("Invalid RunLengthEncoded data", __FILE__, __LINE__);

  // so it knows which swapbytes to call
  uint32_t group_start_index;

  unsigned long low = 0;
  unsigned long num_runs = header_size / header_item_size - 1;
  unsigned long high = num_runs;
  unsigned long mid;
  // low <= x < high

  while (high > low + 1) {
    mid = (high + low) / 2;
    pread(fd, &group_start_index, nByteMode,
	  start + mid * header_item_size + nByteMode);
    if (needConversion && swapBytes) swapbytes(group_start_index);
    
    if (index < group_start_index)
      high = mid; // counts mid out
    else
      low = mid; // doesn't count mid out
  }
   
  ssize_t data_start;
  ssize_t data_end;
  unsigned long group_end_index;
  lseek(fd, start + low * header_item_size, SEEK_SET);
  readSizeType(fd, needConversion, swapBytes, nByteMode, data_start);
  readSizeType(fd, needConversion, swapBytes, nByteMode, group_start_index);
  readSizeType(fd, needConversion, swapBytes, nByteMode, data_end);
  readSizeType(fd, needConversion, swapBytes, nByteMode, group_end_index);
    
  unsigned long group_index = index - group_start_index;
  unsigned long group_length = group_end_index - group_start_index;

  if (group_index >= group_length) {
    // index out of bounds
    ASSERT(low == num_runs - 1);
    std::ostringstream index_str;
    index_str << index << " >= " << group_end_index;
    throw InternalError(string("RunLengthEncoder<T>::seek (index out of bounds, ") + index_str.str() + ")",
                        __FILE__, __LINE__);
  }

  T item;
  if (group_length * sizeof(T) != data_end - data_start) {
    // the group is a run
    lseek(fd, start + data_start, SEEK_SET);
    ::read(fd, &item, sizeof(T));
    if (needConversion && swapBytes) swapbytes(item);
    typename Sequencer::SequenceRule rule;
    if (Sequencer::needRule()) {
      if (data_end - data_start == sizeof(T))
	// must be a default sequence rule
	rule = Sequencer::defaultSequenceRule;
      else {
	::read(fd, &rule, ruleStorageSize(false));
	if (needConversion && swapBytes) swapbytes(rule);
      }
    }
    // rule should be unused below if needRule is false
    return Sequencer::getSequenceItem(item, rule, group_index);
  }
  else {
    // the group is not a run
    pread(fd, &item, sizeof(T), start + data_start + group_index * sizeof(T));
    if (needConversion && swapBytes) swapbytes(item);
    return item;
  }
}

template<class T, class Sequencer>
void RunLengthEncoder<T, Sequencer>::testPrint(std::ostream& out)
{
  finalize();
  unsigned long total_length = 0;
  out << "minimum run length: " << getMinRunLength(false);
  if (getMinRunLength(true) != getMinRunLength(false))
    out << ", " << getMinRunLength(true) << " for default runs.\n";
  else
    out << "\n";
  for (typename std::list<Group>::iterator groupIter = groups_.begin();
       groupIter != groups_.end(); groupIter++) {
    Group& group = *groupIter;
    if (group.isRun()) {
      ASSERT(group.data_.size() == 1);
      out << group.length_ << ": " << group.data_.front();
      if (Sequencer::needRule())
	out << " (" << group.sequenceRule_ << ")\n";
      else
	out << "\n";
    }
    else {
      ASSERT(group.length_ == group.data_.size());
      out << group.length_ << ": ";
      typename std::vector<T>::iterator dataIter = group.data_.begin();
      if (dataIter != group.data_.end())
	out << *dataIter;
      for (dataIter++; dataIter != group.data_.end(); dataIter++)
	out << ", " << *dataIter;
      out << "\n";
    }
    total_length += group.length_;
  }
  ASSERT(total_length == size_);
  out << "Total: " << total_length << "\n";
}

template <class T, class Sequencer>
inline T RunLengthEncoder<T, Sequencer>::iterator::operator*()
{
  const Group& group = *groupIter_;
  if (group.isRun())
    return Sequencer::getSequenceItem(group.data_[0], group.sequenceRule_,
				      (int)groupIndex_);
  else
    return group.data_[groupIndex_];
}

template <class T, class Sequencer>
inline typename RunLengthEncoder<T, Sequencer>::iterator&
RunLengthEncoder<T, Sequencer>::iterator::operator++()
{
  if (groupIndex_ + 1 >= (*groupIter_).length_) {
    groupIter_++;
    groupIndex_ = 0;
  }
  else
    groupIndex_++;
  return *this;
}

template <class T, class Sequencer>
inline typename RunLengthEncoder<T, Sequencer>::iterator&
RunLengthEncoder<T, Sequencer>::iterator::operator--()
{
  if (groupIndex_ == 0) {
    groupIter_--;
    groupIndex_ = (*groupIter_).length_ - 1;
  }
  else
    groupIndex_--;
  return *this;
}

} // End namespace SCIRun

#endif
