/**************************************
 * SuiteTree.h
 * Wayne Witzel
 * SCI Institute,
 * University of Utah
 *
 * Class SuiteTree:
 * Contains a heirarchical tree of testing
 * suites to be summarized and reported.
 *
 ****************************************/

#ifndef SUITE_TREE_H
#define SUITE_TREE_H

#include "Suite.h"
#include <sgi_stl_warnings_off.h>
#include <list>
#include <string>
#include <sgi_stl_warnings_on.h>

using namespace std;

class SuiteTree
{
public:
  SuiteTree() {}
  
  // deletes the whole tree including Suite's
  virtual ~SuiteTree() {}
  
  virtual string composeSummary(const string indent, bool expandAll,
				bool& allPassed) = 0;

  virtual void appendFailedSuites(list<Suite*>& failedSuiteList) = 0;

  virtual void reportAllSuites() = 0;
  
  string summary(bool expandAll)
  { bool dummy; return composeSummary("", expandAll, dummy); }
};

class SuiteTreeNode : public SuiteTree
{
public:
  SuiteTreeNode(string name)
    : myName(name) { }
  
  ~SuiteTreeNode(); // delete sub trees

  void addSubTree(SuiteTree* subTree)
  { mySubTrees.push_back(subTree); }

  SuiteTreeNode* addSubTree(string name)
  { SuiteTreeNode* node = new SuiteTreeNode(name); addSubTree(node);
    return node; }

  inline void addSuite(Suite* suite);

  Suite* addSuite(string name)
  { Suite* suite = new Suite(name); addSuite(suite); return suite; }
  
  string getName()
  { return myName; }
  
  string composeSummary(const string indent, bool expandAll, bool& allPassed);
  string composeSubSummary(const string indent, bool expandAll,
			   bool& allPassed);

  void appendFailedSuites(list<Suite*>& failedSuiteList);

  void reportAllSuites();  
private:
  string myName;
  list<SuiteTree*> mySubTrees;
};

class SuiteTreeLeaf : public SuiteTree
{
public:
  SuiteTreeLeaf(Suite* suite)
    : mySuite(suite) { }

  ~SuiteTreeLeaf()
  { delete mySuite; }
  
  string composeSummary(const string indent, bool expandAll, bool& allPassed);
  
  void appendFailedSuites(list<Suite*>& failedSuiteList);

  void reportAllSuites();  
private:
  Suite* mySuite;
};

inline void SuiteTreeNode::addSuite(Suite* suite)
{
  addSubTree(new SuiteTreeLeaf(suite));
}

#endif // def SUITE_TREE_H
