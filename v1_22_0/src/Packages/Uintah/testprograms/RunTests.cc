/*********************************************
 * RunTests.cc
 * Wayne Witzel
 * SCI Institute,
 * University of Utah
 *
 * Program for running tests and reporting them
 * in a tree of test suites.
 *
 * Add methods to populate this suite tree with more
 * suites below where it says:
 * ADD MORE POPULATING METHODS ABOVE FOR OTHER TEST SUITES
 */

#include "TestSuite/SuiteTree.h"
#include "TestMatrix3/testmatrix3.h"
#include "TestConsecutiveRangeSet/TestConsecutiveRangeSet.h"
#include "TestRangeTree/TestRangeTree.h"
#include "TestBoxGrouper/TestBoxGrouper.h"
#include <stdlib.h>
#include <unistd.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;

void usage(char* prog_name)
{
  cerr << "usage: " << prog_name << " [-e|-a|-h]\n";
  cerr << "\t-e:  expands test suite tree even where all tests have passed\n";
  cerr << "\t-a:  reports all suites (not just failed ones)\n";
  cerr << "\t-v:  verbose mode\n";
  cerr << "\t-h:  lists this help information\n";
}

int main(int argc, char* argv[])
{
  bool expandAll = false;
  bool reportAll = false;
  bool verbose = false;
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] != '-') {
      usage(argv[0]);
      return 1;
    }
    else {
      int j = 1;
      while (argv[i][j] != '\0') {
	switch (argv[i][j]) {
	case 'a':
	  reportAll = true;
	  break;
	case 'e':
	  expandAll = true;
	  break;
	case 'h':
	  usage(argv[0]);
	  return 0;
	case 'v':
	  verbose = true;
	  break;
	default:
	  cerr << "unkown option: " << argv[i][j] << endl;
	  usage(argv[0]);
	  return 1;
	}
	j++;
      }
    }
  }
  
  srand(getpid());
  SuiteTreeNode* suites = new SuiteTreeNode("All Tests");

  // populate the suites tree
  suites->addSubTree(matrix3TestTree());
  suites->addSubTree(ConsecutiveRangeSetTestTree());
  suites->addSubTree(RangeTreeTestTree(verbose, 20000));
  suites->addSubTree(BoxGrouperTestTree(verbose));

  /* ADD MORE POPULATING METHODS ABOVE FOR OTHER TEST SUITES */

  // show suite tree summary
  bool allPassed;
  cout << suites->composeSubSummary("", expandAll, allPassed) << endl;
  
  // report failed suites to itemized the tests the failed
  if (reportAll)
    suites->reportAllSuites();
  else if (!allPassed) {
    list<Suite*> failedSuites;
    suites->appendFailedSuites(failedSuites);
    
    cout << "Failed Suite Reports:" << endl << endl;
    for (list<Suite*>::iterator it = failedSuites.begin();
	 it != failedSuites.end(); it++) {
      (*it)->report();
      cout << endl;
    }

    if (failedSuites.size() == 1)
      cout << "\n1 suite failed.\n";
    else
      cout << endl << failedSuites.size() << " suites failed.\n";
  }

  if (allPassed) {
    cout << "All tests passed!\n";
    return 1;
  }

  return 0;
}

