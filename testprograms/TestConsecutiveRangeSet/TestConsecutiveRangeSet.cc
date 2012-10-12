/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "TestConsecutiveRangeSet.h"
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <sstream>
#include <cstdlib>
#include <cstring>
#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
#endif
#include <climits>

using namespace std;
using namespace SCIRun;

list<int> getRandomList(int size, int min, int max);
void doListInitTests(Suite* suite);
void doAddInOrderTests(Suite* suite);
void doStringInitTests(Suite* suite);
void doIntersectTests(Suite* suite);
void doUnionTests(Suite* suite);

bool equalSets(ConsecutiveRangeSet& set, const int* array, int arraySize,
	       bool& sameSize);
bool equalSets(ConsecutiveRangeSet& set, list<int>& intList, bool& sameSize);

SuiteTree* ConsecutiveRangeSetTestTree()
{
  srand(getpid());  
  SuiteTreeNode* topSuite = new SuiteTreeNode("ConsecutiveRangeSet");

  doListInitTests(topSuite->addSuite("List Init"));
  doAddInOrderTests(topSuite->addSuite("addInOrder"));
  doStringInitTests(topSuite->addSuite("String Init"));
  doIntersectTests(topSuite->addSuite("Intersect"));
  doUnionTests(topSuite->addSuite("Union"));

  return topSuite;
}

list<int> getRandomList(int size, int min, int max)
{
  list<int> result;
  for (int i = 0; i < size; i++)
    result.push_back(rand() % (max - min) + min);
  return result;
}

void doListInitTests(Suite* suite)
{
  int i;
  list<int> intList;
  bool sameSize;
  Test* sizeTest = suite->addTest("Set size");
  Test* compareTest = suite->addTest("Compared items");
  Test* groupTest = suite->addTest("Number of groups");
  
  for (i = 0; i < 100; i++) {
    intList = getRandomList(i, -100, 100);
    ConsecutiveRangeSet set(intList);
    compareTest->setResults(equalSets(set, intList, sameSize));
    sizeTest->setResults(sameSize);
    intList.clear();
  }

  const int testSet[] = { 1, 3, 8, 6, 10, 5, 2, 9, -1 };
  const int sortedTestSet[] = { -1, 1, 2, 3, 5, 6, 8, 9, 10 };
  const int TestSetSize = 9;
  for (i = 0; i < TestSetSize; i++)
    intList.push_back(testSet[i]);
  
  ConsecutiveRangeSet set(intList);
  groupTest->setResults(set.getNumRanges() == 4);  
  compareTest->setResults(equalSets(set, sortedTestSet, TestSetSize,
				    sameSize));
  sizeTest->setResults(sameSize);
}

void doAddInOrderTests(Suite* suite)
{
  int i;
  list<int> intList;
  list<int>::iterator it;
  bool sameSize;
  Test* sizeTest = suite->addTest("Set size");
  Test* compareTest = suite->addTest("Compared items");
  Test* groupTest = suite->addTest("Number of groups");
  
  for (i = 0; i < 100; i++) {
    intList = getRandomList(i, -100, 100);
    intList.sort();
    ConsecutiveRangeSet set;
    for (it = intList.begin(); it != intList.end(); it++)
       set.addInOrder(*it);
    compareTest->setResults(equalSets(set, intList, sameSize));
    sizeTest->setResults(sameSize);
    intList.clear();
  }

  const int testSet[] = { 1, 3, 8, 6, 10, 5, 2, 9, -1 };
  const int sortedTestSet[] = { -1, 1, 2, 3, 5, 6, 8, 9, 10 };
  const int TestSetSize = 9;

  for (i = 0; i < TestSetSize; i++)
    intList.push_back(testSet[i]);
  
  try {
     ConsecutiveRangeSet set;
     for (it = intList.begin(); it != intList.end(); it++)
	set.addInOrder(*it);
     suite->addTest("Not in order", false);
  }
  catch (ConsecutiveRangeSetException) {
     suite->addTest("Not in order", true);
  }

  intList.sort();
  ConsecutiveRangeSet set;
  for (it = intList.begin(); it != intList.end(); it++)
     set.addInOrder(*it);
  groupTest->setResults(set.getNumRanges() == 4);  
  compareTest->setResults(equalSets(set, sortedTestSet, TestSetSize,
				    sameSize));
  sizeTest->setResults(sameSize);
 
}

void doStringInitTest(Suite* suite, string testname, string setstr,
		      const int* expectedset, int expectedset_size,
		      int numgroups, string expectedout);

void doStringInitTests(Suite* suite)
{
  const int normalSet[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 18, 20, 21, 22};
  doStringInitTest(suite, "Normal", "1-8, 10-12, 18, 20-22", normalSet,
		   15, 4, "1 - 8, 10 - 12, 18, 20 - 22");
  doStringInitTest(suite, "Unsorted", "20-22, 10-12, 18, 1-8", normalSet,
		   15, 4, "1 - 8, 10 - 12, 18, 20 - 22" );
  
  const int combineSet[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22};
  doStringInitTest(suite, "Combine", "1-5, 6-10, 19, 20-22", combineSet,
		   14, 2, "1 - 10, 19 - 22");
  doStringInitTest(suite, "Unsorted Combine", "19, 6-10, 1-5, 20-22",
		   combineSet, 14, 2, "1 - 10, 19 - 22");
  doStringInitTest(suite, "Overlapping", "1-6, 3-10, 19-22, 20", combineSet,
		   14, 2, "1 - 10, 19 - 22");
  doStringInitTest(suite, "Unsorted Overlapping", "3-10, 1-6, 19, 22, 19-22",
		   combineSet, 14, 2, "1 - 10, 19 - 22");

  const int negSet[] = {-2, -1};
  doStringInitTest(suite, "Negative", "-1 - -2",
		   negSet, 2, 1, "-2 - -1");
  int one = 1;
  int neg_one = -1;
  doStringInitTest(suite, "Single Negative", "-1",
		   &neg_one, 1, 1, "-1");
  doStringInitTest(suite, "Single Positive", "1",
		   &one, 1, 1, "1");

  Test* exceptionTest = suite->addTest("Parse Exception");
  try {
    ConsecutiveRangeSet set("1-,3-9");
    exceptionTest->setResults(false);
  }
  catch (ConsecutiveRangeSetException) {
    exceptionTest->setResults(true);
  }
  try {
    ConsecutiveRangeSet set("#$%");
    exceptionTest->setResults(false);
  }
  catch (ConsecutiveRangeSetException) {
    exceptionTest->setResults(true);
  }
}


void doStringInitTest(Suite* suite, string testname, string setstr,
		      const int* expectedset, int expectedset_size,
		      int numgroups, string expectedout)
{
  bool sameSize;
  ConsecutiveRangeSet set(setstr);
  Test* test = suite->addTest(testname + ": " + setstr);
  test->setResults(set.getNumRanges() == numgroups);
  test->setResults(equalSets(set, expectedset, expectedset_size, sameSize));
  if (!sameSize) {
    suite->addTest(testname + " size test", false);
  }
  suite->addTest(testname + " output test",
		 strcmp(set.toString().c_str(), expectedout.c_str()) == 0);
}

void doIntersectTests(Suite* suite)
{
  const ConsecutiveRangeSet& empty = ConsecutiveRangeSet::empty;
  const ConsecutiveRangeSet& all = ConsecutiveRangeSet::all;
  ConsecutiveRangeSet testset("1-4, 10-12");
  ConsecutiveRangeSet singlet("-1");
  ConsecutiveRangeSet nonoverlap("5-9");
  ConsecutiveRangeSet overlap("4-10");
  ConsecutiveRangeSet overlap_result("4, 10");
  ConsecutiveRangeSet overlap2("4-13");
  ConsecutiveRangeSet overlap2_result("4, 10-12");

  suite->addTest("with empty", testset.intersected(empty) == empty &&
		 empty.intersected(testset) == empty);
  suite->addTest("with all", testset.intersected(all) == testset &&
		 all.intersected(testset) == testset);  
  suite->addTest("singlet with all", singlet.intersected(all) == singlet &&
		 all.intersected(singlet) == singlet);  
  suite->addTest("all and all", all.intersected(all) == all);  
  suite->addTest("all and ampty", all.intersected(empty) == empty &&
		 empty.intersected(all) == empty);  
  suite->addTest("non-overlap", testset.intersected(nonoverlap) == empty &&
		 nonoverlap.intersected(testset) == empty);  
  suite->addTest("overlap", testset.intersected(overlap) == overlap_result &&
		 overlap.intersected(testset) == overlap_result);  
  suite->addTest("overlap2", testset.intersected(overlap2) == overlap2_result
		 && overlap2.intersected(testset) == overlap2_result);  
}

void doUnionTests(Suite* suite)
{
  const ConsecutiveRangeSet& empty = ConsecutiveRangeSet::empty;
  const ConsecutiveRangeSet& all = ConsecutiveRangeSet::all;
  ConsecutiveRangeSet testset("1-4, 10-12");
  ConsecutiveRangeSet nonoverlap("5-9");
  ConsecutiveRangeSet joined("1-12");
  ConsecutiveRangeSet overlap("4-10");
  ConsecutiveRangeSet overlap2("4-15");
  ConsecutiveRangeSet joined_extended("1-15");
  ConsecutiveRangeSet nonjoin("5, 8, 13, 15-20");
  ConsecutiveRangeSet nonjoined("1-5, 8, 10-13, 15-20");
  ConsecutiveRangeSet singlet("-1");

  suite->addTest("with empty", testset.unioned(empty) == testset &&
		 empty.unioned(testset) == testset);
  suite->addTest("with all", testset.unioned(all) == all &&
		 all.unioned(testset) == all);  
  suite->addTest("all and all", all.unioned(all) == all);
  suite->addTest("all and empty", all.unioned(empty) == all &&
		 empty.unioned(all) == all);  
  suite->addTest("non-overlap", testset.unioned(nonoverlap) == joined &&
		 nonoverlap.unioned(testset) == joined);  
  suite->addTest("overlap", testset.unioned(overlap) == joined &&
		 overlap.unioned(testset) == joined);  
  suite->addTest("overlap2", testset.unioned(overlap2) == joined_extended
		 && overlap2.unioned(testset) == joined_extended);  
  suite->addTest("nonjoined", testset.unioned(nonjoin) == nonjoined
		 && nonjoin.unioned(testset) == nonjoined);
  suite->addTest("same singlet", singlet.unioned(singlet) == singlet);
}

bool equalSets(ConsecutiveRangeSet& set, const int* array, int arraySize,
	       bool& sameSize)
{
  if (arraySize != (int)set.size()) {
    sameSize = false;
    return false;
  }
  sameSize = true;
  
  ConsecutiveRangeSet::iterator it = set.begin();
  for (int i = 0; it != set.end(); it++, i++)
    if (*it != array[i])
      return false;
  return true;
}

bool equalSets(ConsecutiveRangeSet& set, list<int>& intList, bool& sameSize)
{
  intList.sort();
  intList.unique();

  if (intList.size() != set.size()) {
    sameSize = false;
    return false;
  }
  sameSize = true;
  
  ConsecutiveRangeSet::iterator setIt = set.begin();
  list<int>::iterator listIt = intList.begin();
  
  for ( ; setIt != set.end(); setIt++, listIt++)
    if (*setIt != *listIt)
      return false;
  return true;
}

