/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

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

#include <testprograms/TestSuite/uintahshare.h>

using namespace std;

class UINTAHSHARE SuiteTree
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

class UINTAHSHARE SuiteTreeNode : public SuiteTree
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

class UINTAHSHARE SuiteTreeLeaf : public SuiteTree
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
