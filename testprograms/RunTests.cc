#include "TestSuite/SuiteTree.h"
#include "TestMatrix3/testmatrix3.h"
#include <stdlib.h>
#include <unistd.h>

void usage(char* prog_name)
{
  cerr << "usage: " << prog_name << " [-e|-a|-h]\n";
  cerr << "\t-e:  expands test suite tree even where all tests have passed\n";
  cerr << "\t-a:  reports all suites (not just failed ones)\n";
  cerr << "\t-h:  lists this help information\n";
}

int main(int argc, char* argv[])
{
  bool expandAll = false;
  bool reportAll = false;
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

  /*
  Suite* dSuites[5];
  dSuites[0] = new Suite("suite 1");
  dSuites[1] = new Suite("suite 2");
  dSuites[2] = new Suite("suite 3");
  dSuites[3] = new Suite("suite 4");
  dSuites[4] = new Suite("suite 5");

  for (int i = 0; i < 5; i++) {
    dSuites[i]->addTest("Test 1", rand() % 5 > 0);
    dSuites[i]->addTest("Test 2", rand() % 5 > 0);
  }
 
  SuiteTreeNode* t1 = new SuiteTreeNode("Group 1");
  SuiteTreeNode* t2 = new SuiteTreeNode("Sub 1");
  SuiteTreeNode* t3 = new SuiteTreeNode("Sub 2");
  t1->addSubTree(t2);
  t1->addSubTree(t3);
  t2->addSubTree(new SuiteTreeLeaf(dSuites[0]));
  t3->addSubTree(new SuiteTreeLeaf(dSuites[1]));
  SuiteTreeNode* t4 = new SuiteTreeNode("Group 2");
  SuiteTreeNode* t5 = new SuiteTreeNode("Sub 1");
  t4->addSubTree(t5);
  t5->addSubTree(new SuiteTreeLeaf(dSuites[2]));
  SuiteTreeNode* t6 = new SuiteTreeNode("SubSub 1");
  t5->addSubTree(t6);
  t6->addSubTree(new SuiteTreeLeaf(dSuites[3]));
  SuiteTreeNode* t7 = new SuiteTreeNode("Sub 2");
  t4->addSubTree(t7);
  t7->addSubTree(new SuiteTreeLeaf(dSuites[4]));
  suites->addSubTree(t1);
  suites->addSubTree(t4);
  */

  // display summary
  bool allPassed;
  cout << endl << suites->composeSubSummary("", expandAll, allPassed);
  cout << endl;
  
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

