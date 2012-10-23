/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
#include <list>
#include <string>


class SuiteTree
{
public:
  SuiteTree() {}
  
  // deletes the whole tree including Suite's
  virtual ~SuiteTree() {}
  
  virtual std::string composeSummary(const std::string indent, bool expandAll,
				bool& allPassed) = 0;

  virtual void appendFailedSuites(std::list<Suite*>& failedSuiteList) = 0;

  virtual void reportAllSuites() = 0;
  
  std::string summary(bool expandAll)
  { bool dummy; return composeSummary("", expandAll, dummy); }
};

class SuiteTreeNode : public SuiteTree
{
public:
  SuiteTreeNode(std::string name)
    : myName(name) { }
  
  ~SuiteTreeNode(); // delete sub trees

  void addSubTree(SuiteTree* subTree)
  { mySubTrees.push_back(subTree); }

  SuiteTreeNode* addSubTree(std::string name)
  { SuiteTreeNode* node = new SuiteTreeNode(name); addSubTree(node);
    return node; }

  inline void addSuite(Suite* suite);

  Suite* addSuite(std::string name)
  { Suite* suite = new Suite(name); addSuite(suite); return suite; }
  
  std::string getName()
  { return myName; }
  
  std::string composeSummary(const std::string indent, bool expandAll, bool& allPassed);
  std::string composeSubSummary(const std::string indent, bool expandAll,
			   bool& allPassed);

  void appendFailedSuites(std::list<Suite*>& failedSuiteList);

  void reportAllSuites();  
private:
  std::string myName;
  std::list<SuiteTree*> mySubTrees;
};

class SuiteTreeLeaf : public SuiteTree
{
public:
  SuiteTreeLeaf(Suite* suite)
    : mySuite(suite) { }

  ~SuiteTreeLeaf()
  { delete mySuite; }
  
  std::string composeSummary(const std::string indent, bool expandAll, bool& allPassed);
  
  void appendFailedSuites(std::list<Suite*>& failedSuiteList);

  void reportAllSuites();  
private:
  Suite* mySuite;
};

inline void SuiteTreeNode::addSuite(Suite* suite)
{
  addSubTree(new SuiteTreeLeaf(suite));
}

#endif // def SUITE_TREE_H
