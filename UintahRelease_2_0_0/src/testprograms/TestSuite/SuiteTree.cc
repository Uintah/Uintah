/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include "SuiteTree.h"
#include <iostream>

SuiteTreeNode::~SuiteTreeNode()
{
  for (std::list<SuiteTree*>::iterator it = mySubTrees.begin();
       it != mySubTrees.end() ; it++)
    delete *it;
}

std::string SuiteTreeNode::composeSubSummary(const std::string indent, bool expandAll,
					bool& allPassed)
{
  std::string summary = "";
  allPassed = true;
  bool passed;
  for (std::list<SuiteTree*>::iterator it = mySubTrees.begin();
       it != mySubTrees.end() ; it++) {
    summary += (*it)->composeSummary(indent, expandAll, passed);
    if (!passed)
      allPassed = false;
  }
  return summary;
}

std::string SuiteTreeNode::composeSummary(const std::string indent, bool expandAll,
				     bool& allPassed)
{
  std::string summary = composeSubSummary(indent + "  ", expandAll, allPassed);
 
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

void SuiteTreeNode::appendFailedSuites(std::list<Suite*>& failedSuiteList)
{
  for (std::list<SuiteTree*>::iterator it = mySubTrees.begin();
       it != mySubTrees.end() ; it++)
    (*it)->appendFailedSuites(failedSuiteList);
}

void SuiteTreeNode::reportAllSuites()
{
  for (std::list<SuiteTree*>::iterator it = mySubTrees.begin();
       it != mySubTrees.end() ; it++)
    (*it)->reportAllSuites();  
}

std::string SuiteTreeLeaf::composeSummary(const std::string indent, bool,
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

void SuiteTreeLeaf::appendFailedSuites(std::list<Suite*>& failedSuiteList)
{
  if (!mySuite->hasAllPassed())
    failedSuiteList.push_back(mySuite);
}

void SuiteTreeLeaf::reportAllSuites()
{
  mySuite->report();
  std::cout << std::endl;
}

