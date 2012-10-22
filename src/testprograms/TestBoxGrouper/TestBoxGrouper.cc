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

#include <testprograms/TestBoxGrouper/TestBoxGrouper.h>
#include <testprograms/TestBoxGrouper/Box.h>
#include <testprograms/TestBoxGrouper/BoxRangeQuerier.h>
#include <Core/Containers/SuperBox.h>

using namespace std;
using namespace SCIRun;

namespace Uintah {

typedef SuperBox<const Box*, IntVector, int, int,
  InternalAreaSuperBoxEvaluator<const Box*, int> > SuperBox;
typedef SuperBoxSet<const Box*, IntVector, int, int,
  InternalAreaSuperBoxEvaluator<const Box*, int> > SuperBoxSet;

void doSimpleExampleTests(Suite* suite, bool verbose);
void doGridTests(Suite* suite, int n, int numTakeAway, bool verbose);

// test if superBoxSet is null, and that superBoxes in the superBoxSet
// are disjoint and complete (with respect to boxes).
void performStandardSuperBoxSetTests(Suite* suite,
				     const SuperBoxSet* superBoxSet,
				     const set<const Box*>& boxes);

template <class BoxPIterator>
void clean(BoxPIterator boxesBegin, BoxPIterator boxesEnd);

SuiteTree* BoxGrouperTestTree(bool verbose)
{  
  SuiteTreeNode* topSuite = new SuiteTreeNode("BoxGrouper");

  doSimpleExampleTests(topSuite->addSuite("Simple Example"), verbose);
  doGridTests(topSuite->addSuite("nxnxn Grid"), 4, 0, verbose);
  doGridTests(topSuite->addSuite("nxnxn Grid - 1"), 4, 1, verbose);  
  doGridTests(topSuite->addSuite("nxnxn Grid - 2"), 4, 2, verbose);
  doGridTests(topSuite->addSuite("nxnxn Grid - 3"), 4, 3, verbose);
  return topSuite;
}

void doSimpleExampleTests(Suite* suite, bool verbose)
{
  // just some example I drew on my whiteboard
  set<const Box*> boxes;
  boxes.insert(scinew Box(IntVector(6, 1, 0), IntVector(15, 5, 0), 1));
  boxes.insert(scinew Box(IntVector(6, 6, 0), IntVector(10, 10, 0), 2));
  boxes.insert(scinew Box(IntVector(11, 6, 0), IntVector(15, 15, 0), 3));
  boxes.insert(scinew Box(IntVector(6, 11, 0), IntVector(10, 15, 0), 4));
  boxes.insert(scinew Box(IntVector(1, 6, 0), IntVector(5, 10, 0), 5));
  boxes.insert(scinew Box(IntVector(1, 11, 0), IntVector(5, 20, 0), 6));
  boxes.insert(scinew Box(IntVector(6, 16, 0), IntVector(15, 20, 0), 7));  
  boxes.insert(scinew Box(IntVector(16, 1, 0), IntVector(20, 5, 0), 8));

  BoxRangeQuerier rangeQuerier(boxes.begin(), boxes.end());
  SuperBoxSet* superBoxSet =
    SuperBoxSet::makeOptimalSuperBoxSet(boxes.begin(), boxes.end(),
					rangeQuerier);
  if (verbose) {
    cerr << "\nSuperBoxSet:\n";
    cerr << *superBoxSet << endl;
    cerr << superBoxSet->getValue() << endl;
  }
  
  performStandardSuperBoxSetTests(suite, superBoxSet, boxes);
  suite->addTest("Value", superBoxSet->getValue() == 50);

  clean(boxes.begin(), boxes.end());
}

void doGridTests(Suite* suite, int n, int numTakeAway, bool verbose)
{
  list<int> takeAwayRands;
  int i;
  for (i = 0; i < numTakeAway; i++) {
    takeAwayRands.push_back(rand() % (n*n*n - i));
  }
  takeAwayRands.sort();
  
  list<int>::iterator takeAwayRandIter = takeAwayRands.begin();
  
  set<const Box*> boxes;
  i = 0;
  for (int x = 1; x <= n; x++) {
    for (int y = 1; y <= n; y++) {
      for (int z = 1; z <= n; z++) {
	int id;
	if (n < 10)
	  id = x * 100 + y * 10 + z;
	else
	  id = i;
	if (takeAwayRandIter == takeAwayRands.end() ||
	    i < *takeAwayRandIter) {
	  boxes.insert(scinew Box(IntVector(x, y, z), IntVector(x, y, z), id));
	}
	else {
	  ++takeAwayRandIter;
	}
	i++;
      }
    }
  }
  suite->addTest("take away count", (int)boxes.size() == n*n*n - numTakeAway);
  
  //boxes.erase(boxes.begin());

  BoxRangeQuerier rangeQuerier(boxes.begin(), boxes.end());

#ifdef SUPERBOX_PERFORMANCE_TESTING
  SuperBoxSet::biggerBoxCount = 0;
  SuperBoxSet::minBiggerBoxCount = 0;
#endif
  SuperBoxSet* superBoxSet =
    SuperBoxSet::makeOptimalSuperBoxSet(boxes.begin(), boxes.end(),
					    rangeQuerier);
  if (verbose) {
#ifdef SUPERBOX_PERFORMANCE_TESTING  
    cerr << "\nBiggerBoxCount: " << SuperBoxSet::biggerBoxCount << endl;
    cerr << "\nMinimum BiggerBoxCount: " << SuperBoxSet::minBiggerBoxCount
	 << endl;
#endif  
    cerr << "\nOptimal SuperBoxSet:\n";
    cerr << *superBoxSet << endl;
    cerr << superBoxSet->getValue() << endl;
  }
  
  performStandardSuperBoxSetTests(suite, superBoxSet, boxes);
  if (numTakeAway == 0) {
    suite->addTest("Value", superBoxSet->getValue() == 3*n*n*(n-1));
  }
  else {
    SuperBoxSet* nearOptimalSuperBoxSet =
      SuperBoxSet::makeNearOptimalSuperBoxSet(boxes.begin(), boxes.end(),
					      rangeQuerier);
    if (verbose) {
      cerr << "\nNear Optimal (heuristic) SuperBoxSet:\n";
      cerr << *nearOptimalSuperBoxSet << endl;
      cerr << nearOptimalSuperBoxSet->getValue() << endl;
    }
    
    suite->addTest("Near Optimal Comparison", superBoxSet->getValue() >=
		   nearOptimalSuperBoxSet->getValue());
  }
  
  clean(boxes.begin(), boxes.end());
}

void performStandardSuperBoxSetTests(Suite* suite,
				     const SuperBoxSet* superBoxSet,
				     const set<const Box*>& boxes)
{
  suite->addTest("not null", superBoxSet != 0);
  if (superBoxSet == 0)
    return;

  set<const Box*> superSetBoxes;
  int count = 0;
  vector<SuperBox*>::const_iterator iter;
  for (iter = superBoxSet->getSuperBoxes().begin();
       iter != superBoxSet->getSuperBoxes().end(); iter++) {
    const vector<const Box*>& superBoxBoxes = (*iter)->getBoxes();
    superSetBoxes.insert(superBoxBoxes.begin(), superBoxBoxes.end());
    count += superBoxBoxes.size();    
  }

  suite->addTest("complete", superSetBoxes == boxes);
  suite->addTest("disjoint", count == (int)superSetBoxes.size());
}

template <class BoxPIterator>
void clean(BoxPIterator boxesBegin, BoxPIterator boxesEnd)
{
  for (BoxPIterator iter = boxesBegin; iter != boxesEnd; iter++) {
    delete *iter;
  }
}
  
} // end namespace Uintah

