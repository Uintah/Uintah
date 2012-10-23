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

#include "TestRangeTree.h"
#include <Core/Containers/RangeTree.h>
#include "Point.h"
#include <cstdlib>
#ifndef _WIN32
#include <unistd.h>
#include <sys/time.h>
#endif
#include <cmath>
#include <iostream>
#include <cstring>

using namespace std;
using namespace SCIRun;

list<Point*> getRandomPoints(int n);
list<Point*> doLinearQuery(list<Point*> points,
			   const Point& low, const Point& high);
list<Point*> doLinearSphereQuery(list<Point*> points,
				 const Point& p, int radius);
void doRangeQueryTests(Suite* suite, RangeTree<Point, int>* rangeTree,
		       list<Point*>& points,
		       const Point& low, const Point& high, bool verbose);
void doSphereRangeQueryTests(Suite* suite, RangeTree<Point, int>* rangeTree,
			     list<Point*>& points,
			     const Point& p, int radius, bool verbose);
Point* doLinearNearestL1Query(list<Point*> points, const Point& p);
Point* doLinearNearestQuery(list<Point*> points, const Point& p);

void doNearestQueryTests(Suite* suite, RangeTree<Point, int, true>* rangeTree,
			 list<Point*>& points, const Point& p, bool verbose);
void doNearestQueryAtPointsTest(Suite* suite,
				RangeTree<Point, int, true>* rangeTree,
				list<Point*>& points, bool verbose);

void printList(list<Point*>& points);

void display_time_diff(ostream& out, timeval start, timeval end,
		       unsigned long divisor = 1)
{
  if ((start.tv_sec - end.tv_sec) / divisor == 0) {
    out << (end.tv_usec - start.tv_usec) / divisor << " microseconds";
  }
  else {
    double sec = ((double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000) / (double)divisor;
    if (sec < 1)
      out << (int)(sec * 1000000 + 0.5) << " microseconds";
    else
      out << sec << " seconds";
  }
}

SuiteTree* RangeTreeTestTree(bool verbose /*= false*/, int n /* = 100000 */,
			     int randomSeed /*= 0*/)
{
  SuiteTreeNode* topSuite = new SuiteTreeNode("RangeTree");

  if (randomSeed == 0)
    randomSeed = getpid();
  srand(randomSeed);

  if (verbose)
    cout << "Using random number seed " << randomSeed << endl;
  
  list<Point*> points;
  points = getRandomPoints(n);
  points.push_back(new Point(-1, 100, 100, 100));
  points.push_back(new Point(-2, 100, 100, 101));
  points.push_back(new Point(-3, 100, 100, 99));
  points.push_back(new Point(-4, 100, 101, 100));
  points.push_back(new Point(-5, 100, 99, 100));
  points.push_back(new Point(-7, 100, 0, 0));
  points.push_back(new Point(-8, 0, 0, 100));
  points.push_back(new Point(-10, 0, 0, 0));
  points.push_back(new Point(-20, 0, 0, 1));
  points.push_back(new Point(-30, 0, 0, -1));
  points.push_back(new Point(-40, 0, 1, 0));
  points.push_back(new Point(-50, 0, -1, 0));
  points.push_back(new Point(-11, 1, 0, 0));
  points.push_back(new Point(-21, 1, 0, 1));
  points.push_back(new Point(-31, 1, 0, -1));
  points.push_back(new Point(-41, 1, 1, 0));
  points.push_back(new Point(-51, 1, -1, 0));

  for (int i = 0; i < 2; i++) {
    // duplicate these to test duplicates
    points.push_back(new Point(-12 - i * 100, 50, 100, 50));
    points.push_back(new Point(-22 - i * 100, 50, 50, 101));
    points.push_back(new Point(-32 - i * 100, 50, 50, -1));
    points.push_back(new Point(-42 - i * 100, 100, 50, 51));
    points.push_back(new Point(-52 - i * 100, 49, 0, 50));
  }
 
  /*
    points.push_back(new Point(-12, -1, 0, 0));
    points.push_back(new Point(-22, -1, 0, 1));
    points.push_back(new Point(-32, -1, 0, -1));
    points.push_back(new Point(-42, -1, 1, 0));
    points.push_back(new Point(-52, -1, -1, 0));
  */
  Point low(-1, 0, 0, 0);
  Point high(-1, 100, 100, 100);
  Point p(-1, 50, 50, 50);
  int radius = 50;
  
  timeval start, end;
  gettimeofday(&start, 0);
  RangeTree<Point, int>* rangeTree = scinew RangeTree<Point, int>(points, 3);
  gettimeofday(&end, 0);
  if (verbose) {
    cout << "Built tree in ";
    display_time_diff(cout, start, end);
    cout << ".\n\n";
    //rangeTree->topLevelDump();
    //rangeTree->bottomLevelDump();
    //cout << endl;
  }

  
  doRangeQueryTests(topSuite->addSuite("Range Query"), rangeTree, points,
		    low, high, verbose);
  if (verbose) cout << endl;
  doSphereRangeQueryTests(topSuite->addSuite("Sphere Range Query"), rangeTree,
			  points, p, radius, verbose); 
  if (verbose) cout << endl;


  delete rangeTree;
  rangeTree = 0;
  
  // build a range tree for nearest neighbor searches
  gettimeofday(&start, 0);
  RangeTree<Point, int, true>* nearestCapabableRangeTree =
    scinew RangeTree<Point, int, true>(points, 3);
  gettimeofday(&end, 0);
  if (verbose) {
    cout << "Built nearest query enabled tree in ";
    display_time_diff(cout, start, end);
    cout << ".\n\n";
    //rangeTree->topLevelDump();
    //rangeTree->bottomLevelDump();
    //cout << endl;
  }

  Point p_out[8];
  p_out[0] = Point(-1, 500, 500, 500);
  p_out[1] = Point(-1, -500, -500, -500);
  p_out[2] = Point(-1, -500, -500, 500);
  p_out[3] = Point(-1, 500, -500, -500);
  p_out[4] = Point(-1, -500, 500, -500);
  p_out[5] = Point(-1, -500, 500, 500);
  p_out[6] = Point(-1, 500, -500, 500);
  p_out[7] = Point(-1, 500, 500, -500);
  Point p_in(-1, 50, 50, 50);
  SuiteTreeNode* nearestSuite = topSuite->addSubTree("Nearest Neighbor Query");
  char outerSuiteName[14];
  strcpy(outerSuiteName, "Outer point ");
  outerSuiteName[13] = '\0';

  for (int i = 0; i < 8; i++) {
    outerSuiteName[12] = '0' + i;
    doNearestQueryTests(nearestSuite->addSuite(outerSuiteName),
			nearestCapabableRangeTree, points, p_out[i], verbose);
    if (verbose) cout << endl;
  }
  Suite* innerPointNearestSuite = nearestSuite->addSuite("Inner point");
  doNearestQueryTests(innerPointNearestSuite, nearestCapabableRangeTree,
		      points, p_in, verbose);
  if (verbose) cout << endl;
  doNearestQueryAtPointsTest(innerPointNearestSuite, nearestCapabableRangeTree,
			     points, verbose); 
  if (verbose) cout << endl;
  delete rangeTree;
  
  // clean up 
  for (list<Point*>::iterator it = points.begin(); it != points.end(); it++)
    delete *it;

  return topSuite;
} 

void doRangeQueryTests(Suite* suite, RangeTree<Point, int>* rangeTree,
		       list<Point*>& points,
		       const Point& low, const Point& high, bool verbose)
{
  timeval start, end;
  list<Point*> treeQuery;
  // do the query  twice and time the second one in attempts to get a more
  // accurate timing apart from memory delays that may play more of a role
  // the first time
  rangeTree->query(low, high, treeQuery);
  if (verbose) {
    treeQuery.clear();
    gettimeofday(&start, 0);
    rangeTree->query(low, high, treeQuery);
    gettimeofday(&end, 0);
  }
  
  if (verbose) {
    cout << "Tree query found " << treeQuery.size() << " out of " << points.size() << " points in ";
    display_time_diff(cout, start, end);
    cout << ".\n";
  }
  
  gettimeofday(&start, 0);
  list<Point*> linearQuery = doLinearQuery(points, low, high);
  gettimeofday(&end, 0);
  if (verbose) {
    cout << "Linear query found " << linearQuery.size() << " out of " << points.size() << " points in ";
    display_time_diff(cout, start, end);
    cout << ".\n";
  }
  
  treeQuery.sort();
  linearQuery.sort();
  
  bool correctCount = (treeQuery.size() == linearQuery.size());
  suite->addTest("Range query count", correctCount);

  if (correctCount)
  {
    list<Point*>::iterator treeResultIt = treeQuery.begin();
    list<Point*>::iterator linResultIt = linearQuery.begin();
    int i = 0;
    bool same = true;
    while (treeResultIt != treeQuery.end()) {
      if (*treeResultIt != *linResultIt) {
	if (verbose)
	  cout << "Element " << i << " is not the same.\n";
	same = false;
      }
      
      treeResultIt++;
      linResultIt++;
      i++;
    }

    suite->addTest("Range query", same);    
  }
  else if (verbose && points.size() < 50) {
    cout << "Range Tree query results: " << endl;
    printList(treeQuery);
    cout << "Linear query results: " << endl;
    printList(linearQuery);    
  }
}

void doSphereRangeQueryTests(Suite* suite, RangeTree<Point, int>* rangeTree,
			     list<Point*>& points,
			     const Point& p, int radius, bool verbose)
{   
  timeval start, end;
  list<Point*> treeQuery;
  // do the query  twice and time the second one in attempts to get a more
  // accurate timing apart from memory delays that may play more of a role
  // the first time
  rangeTree->querySphere(p, radius, treeQuery);
  if (verbose) {
    treeQuery.clear();
    gettimeofday(&start, 0);
    //rangeTree->querySphereNonStrict(p, radius, treeQuery);
    rangeTree->querySphere(p, radius, treeQuery);
    gettimeofday(&end, 0);
  }
  
  if (verbose) {
    cout << "Tree sphere query found " << treeQuery.size() << " out of " << points.size() << " points in ";
    display_time_diff(cout, start, end);
    cout << ".\n";
  }
  
  gettimeofday(&start, 0);
  list<Point*> linearQuery = doLinearSphereQuery(points, p, radius);
  gettimeofday(&end, 0);
  if (verbose) {
    cout << "Linear sphere query found " << linearQuery.size() << " out of " << points.size() << " points in ";
    display_time_diff(cout, start, end);
    cout << ".\n";
  }
  
  treeQuery.sort();
  linearQuery.sort();
  
  bool correctCount = (treeQuery.size() == linearQuery.size());
  suite->addTest("Range query count", correctCount);

  if (correctCount)
  {
    list<Point*>::iterator treeResultIt = treeQuery.begin();
    list<Point*>::iterator linResultIt = linearQuery.begin();
    int i = 0;
    bool same = true;
    while (treeResultIt != treeQuery.end()) {
      if (*treeResultIt != *linResultIt) {
	if (verbose)
	  cout << "Element " << i << " is not the same.\n";
	same = false;
      }
      
      treeResultIt++;
      linResultIt++;
      i++;
    }

    suite->addTest("Range query", same);    
  }
  else if (verbose && points.size() < 50) {
    cout << "Range Tree sphere query results: " << endl;
    printList(treeQuery);
    cout << "Linear sphere query results: " << endl;
    printList(linearQuery);    
  }
}

void doNearestQueryTests(Suite* suite, RangeTree<Point, int, true>* rangeTree,
			 list<Point*>& points, const Point& p, bool verbose)
{
  timeval start, end;
  // do it twice and time the second one in attempts to get a more accurate
  // timing apart from memory delays that may play more of a role the first
  // time
  Point* nearestL1 = rangeTree->queryNearestL1(p, INT_MAX/2);
  if (verbose) {
    gettimeofday(&start, 0);
    nearestL1 = rangeTree->queryNearestL1(p, INT_MAX/2);
    gettimeofday(&end, 0);
  }

  if (verbose) {
    cout << "Tree nearest L1 query done in ";
    display_time_diff(cout, start, end);
    cout << ".\n";
  }

  // do it twice and time the second one in attempts to get a more accurate
  // timing apart from memory delays that may play more of a role the first
  // time
  Point* nearest = rangeTree->queryNearest(p, INT_MAX/2);
  if (verbose) {
    gettimeofday(&start, 0);
    nearest = rangeTree->queryNearest(p, INT_MAX/2);
    gettimeofday(&end, 0);
  }
  
  if (verbose) {
    cout << "Tree nearest query done in ";
    display_time_diff(cout, start, end);
    cout << ".\n";
  }
  
  gettimeofday(&start, 0);
  Point* linearNearestL1 = doLinearNearestL1Query(points, p);
  gettimeofday(&end, 0);
  if (verbose) {
    cout << "Linear nearest L1 query done in ";
    display_time_diff(cout, start, end);
    cout << ".\n";
  }

  gettimeofday(&start, 0);
  Point* linearNearest = doLinearNearestQuery(points, p);
  gettimeofday(&end, 0);
  if (verbose) {
    cout << "Linear nearest query done in ";
    display_time_diff(cout, start, end);
    cout << ".\n";
  }

  int rangeDistL1 = p.distanceL1(*nearestL1);
  int linearDistL1 = p.distanceL1(*linearNearestL1);
  if (verbose && (rangeDistL1 != linearDistL1)) 
    cout << "Failed Nearest L1: " << rangeDistL1 << " != " << linearDistL1 << endl;
  suite->addTest("Nearest L1 query", rangeDistL1 == linearDistL1);
  int rangeDistSqrd = p.distanceSquared(*nearest);
  int linearDistSqrd = p.distanceSquared(*linearNearest);
  suite->addTest("Nearest query", rangeDistSqrd == linearDistSqrd);
}

void doNearestQueryAtPointsTest(Suite* suite,
				RangeTree<Point, int, true>* rangeTree,
				list<Point*>& points, bool verbose)
{ 
  timeval start, end;
  // now do nearest queries where the query point is one of the points in
  // the tree
  list<Point*>::iterator pIter = points.begin();
  bool passed = true;
  gettimeofday(&start, 0);  
  for ( ; pIter != points.end(); pIter++) {
    Point* nearest = rangeTree->queryNearest(*(*pIter), INT_MAX/2);
    if (nearest->distanceL1(*(*pIter)) != 0) {
      passed = false;
      cout << nearest->getId() << " not @ " << (*pIter)->getId() << endl;
    }
  }
  gettimeofday(&end, 0);
  suite->findOrAddTest("@ Nearest", passed);

  if (verbose) {
    cout << "Tree nearest query @ point done in ";
    display_time_diff(cout, start, end, points.size());
    cout << " average time.\n";
  }  
}

  
  
list<Point*> getRandomPoints(int n)
{
  list<Point*> points;
  for (int i = 0; i < n; i++)
    points.push_back(new Point(i, rand() % 500 - 250, rand() % 500 - 250,
			       rand() % 500 - 250));
  return points;
}

list<Point*> doLinearQuery(list<Point*> points,
			   const Point& low, const Point& high)
{
  list<Point*> found;
  int i;
  for (list<Point*>::iterator it = points.begin(); it != points.end(); it++) {
    Point* p = *it;
    for (i = 0; i < 3; i++) {
      if ((*p)[i] < low[i] || (*p)[i] > high[i]) {
	break;
      }
    }
    if (i == 3)
      found.push_back(*it);
  }
  return found;
}

list<Point*> doLinearSphereQuery(list<Point*> points,
				 const Point& p, int radius)
{
  list<Point*> found;
  int i;
  int radiusSquared = radius * radius;
  for (list<Point*>::iterator it = points.begin(); it != points.end(); it++) {
    Point* p2 = *it;
    int distSquared = 0;
    for (i = 0; i < 3; i++) {
      distSquared += ((*p2)[i] - p[i]) * ((*p2)[i] - p[i]);
    }
    if (distSquared <= radiusSquared)
      found.push_back(*it);
  }
  return found;
}

Point* doLinearNearestL1Query(list<Point*> points, const Point& p)
{
  Point* found = 0;
  int minDistance = INT_MAX;
  
  for (list<Point*>::iterator it = points.begin(); it != points.end(); it++) {
    Point* p2 = *it;
    int distance = p.distanceL1(*p2);
    if (distance < minDistance) {
      minDistance = distance;
      found = p2;
    }
  }
  return found;
}

Point* doLinearNearestQuery(list<Point*> points, const Point& p)
{
  Point* found = 0;
  int minDistanceSquared = INT_MAX;
  
  for (list<Point*>::iterator it = points.begin(); it != points.end(); it++) {
    Point* p2 = *it;
    int distanceSquared = p.distanceSquared(*p2);
    if (distanceSquared < minDistanceSquared) {
      minDistanceSquared = distanceSquared;
      found = p2;
    }
  }
  return found;
}

void printList(list<Point*>& points)
{
  for (list<Point*>::iterator it = points.begin(); it != points.end(); it++)
    cout << (**it)[0] << ", " << (**it)[1] << ", " << (**it)[2] << endl;

}
