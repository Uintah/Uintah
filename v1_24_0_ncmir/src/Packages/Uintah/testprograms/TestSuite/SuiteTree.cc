#include "SuiteTree.h"
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
using namespace std;

SuiteTreeNode::~SuiteTreeNode()
{
  for (list<SuiteTree*>::iterator it = mySubTrees.begin();
       it != mySubTrees.end() ; it++)
    delete *it;
}

string SuiteTreeNode::composeSubSummary(const string indent, bool expandAll,
					bool& allPassed)
{
  string summary = "";
  allPassed = true;
  bool passed;
  for (list<SuiteTree*>::iterator it = mySubTrees.begin();
       it != mySubTrees.end() ; it++) {
    summary += (*it)->composeSummary(indent, expandAll, passed);
    if (!passed)
      allPassed = false;
  }
  return summary;
}

string SuiteTreeNode::composeSummary(const string indent, bool expandAll,
				     bool& allPassed)
{
  string summary = composeSubSummary(indent + "  ", expandAll, allPassed);
 
  if (allPassed) {
    if (expandAll)
      summary = indent + "P " + getName() + "\n" + summary;
    else
      summary = indent + "P " + getName() + "\n";      
  }
  else
    summary = indent + "F " + getName() + "\n" + summary;

  return summary;
}

void SuiteTreeNode::appendFailedSuites(list<Suite*>& failedSuiteList)
{
  for (list<SuiteTree*>::iterator it = mySubTrees.begin();
       it != mySubTrees.end() ; it++)
    (*it)->appendFailedSuites(failedSuiteList);
}

void SuiteTreeNode::reportAllSuites()
{
  for (list<SuiteTree*>::iterator it = mySubTrees.begin();
       it != mySubTrees.end() ; it++)
    (*it)->reportAllSuites();  
}

string SuiteTreeLeaf::composeSummary(const string indent, bool,
				     bool& allPassed)
{
  if (mySuite->hasAllPassed()) {
    allPassed = true;
    return indent + "P " + mySuite->getName() + '\n';
  }
  else {
    allPassed = false;
    return indent + "F " + mySuite->getName() + '\n';
  }
}

void SuiteTreeLeaf::appendFailedSuites(list<Suite*>& failedSuiteList)
{
  if (!mySuite->hasAllPassed())
    failedSuiteList.push_back(mySuite);
}

void SuiteTreeLeaf::reportAllSuites()
{
  mySuite->report();
  cout << endl;
}

