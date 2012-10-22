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

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "GV_TaskGraph.h"
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>

using namespace std;
using namespace SCIRun;
using namespace Uintah;

float safePercent(double num, double denom)
{ return (denom != 0) ? num / denom: 0; }

GV_Task::GV_Task(string name, double duration, GV_TaskGraph* owner)
  : m_name(name), m_duration(duration),
    m_graph(owner),
    m_maxBelowCost(0), m_maxAboveCost(0),
    m_sorted(false),m_visited(false)
{}

GV_Task::~GV_Task()
{}

float GV_Task::getMaxPathPercent() const
{
  return safePercent(getMaxInclusivePathCost(),
		     m_graph->getCriticalPathCost());
}

Edge* GV_Task::addDependency(GV_Task* task)
{
  for (list<Edge*>::const_iterator iter = m_dependencyEdges.begin();
       iter != m_dependencyEdges.end(); iter++) {
    if (task == (*iter)->getSource())
      return 0;
  }

  Edge* newEdge = scinew Edge(task, this);
  
  m_dependencyEdges.push_back(newEdge);
  task->m_dependentEdges.push_back(newEdge); 
  return newEdge;
}

Edge::Edge(GV_Task* dependency, GV_Task* dependent)
  : m_source(dependency), m_target(dependent), m_obsolete(false)
{}

float Edge::getMaxPathPercent() const
{
  return safePercent(getMaxInclusivePathCost(),
		     getGraph()->getCriticalPathCost());
}

void Edge::relaxEdgeUp()
{
  m_source->testSetMaxBelowCost(m_target->getMaxInclBelowCost());
}

void Edge::relaxEdgeDown()
{
  m_target->testSetMaxAboveCost(m_source->getMaxInclAboveCost());
} 


GV_TaskGraph*
GV_TaskGraph::inflate(string xmlDir)
{
//    try {
//      XMLPlatformUtils::Initialize();
//    } catch (const XMLException& e) {
//      cerr << "Unable to initialize XML library: " << e.getMessage() << endl;
//      return 0;
//    }

  list<ProblemSpecP> docs;

  GV_TaskGraph* pGraph = scinew GV_TaskGraph();
  
  int process = 0;
  string xmlFileName;
  FILE* tstFile;
  do {
    ostringstream pname;
    pname << "/taskgraph_" << setw(5) << setfill('0') << process << ".xml";
    xmlFileName = xmlDir + pname.str();
    
    if ((tstFile = fopen(xmlFileName.c_str(), "r")) == NULL)
      break;
    fclose(tstFile);

    ProblemSpecP prob_spec = ProblemSpecReader().readInputFile( xmlFileName );
 
    docs.push_back(prob_spec);

    pGraph->readNodes(prob_spec);
    process++;
  } while (process < 100000 /* it will most likely always break out of loop
			       -- but just so it won't ever be caught in an
			       infinite loop */);  
  if (process == 0) {
    cerr << "Task graph data does not exist:" << endl;
    cerr << xmlFileName << " does not exist." << endl;
    delete pGraph;
    return 0;
  }
  
  for (list<ProblemSpecP>::iterator docIter = docs.begin();
       docIter != docs.end(); docIter++) {
    pGraph->readEdges(*docIter);
  }
  pGraph->computeMaxPathLengths();
  
  return pGraph;
}

GV_TaskGraph::GV_TaskGraph()
  : m_criticalPathCost(0), m_thresholdPercent(0)
{
}

void GV_TaskGraph::readNodes(ProblemSpecP xmlDoc)
{
  ProblemSpecP nodes = xmlDoc->findBlock("Nodes");
  for (ProblemSpecP node = nodes->findBlock("node"); node != 0;
       node = node->findNextBlock("node")) {
    string task_name;
    double task_duration;
    node->get("name", task_name);
    node->get("duration", task_duration);
    
    GV_Task* task;
    if ((task = findTask(task_name)) != NULL) {
      // task already exists
      // It may be a reduction task... in any case
      // make its duration the maximum of given durations.
      task->testSetDuration(task_duration); 
    }
    else {
      task = scinew GV_Task(task_name, task_duration, this);
      m_tasks.push_back(task);
      m_taskMap[task_name] = task;
    }
  }
}

void GV_TaskGraph::readEdges(ProblemSpecP xmlDoc)
{
  ProblemSpecP edges = xmlDoc->findBlock("Edges");
  for (ProblemSpecP node = edges->findBlock("edge"); node != 0;
       node = node->findNextBlock("edge")) {
    string source;
    string target;
    node->get("source", source);
    node->get("target", target);
    GV_Task* sourceTask = m_taskMap[source];
    GV_Task* targetTask = m_taskMap[target];

    if (sourceTask != NULL && targetTask != NULL) {
      if (m_edgeMap.find(source + " -> " + target) == m_edgeMap.end()) {
	Edge* edge = targetTask->addDependency(sourceTask);
	if (edge) {
	  m_edgeMap[source + " -> " + target] = edge;
	}
      }
    }
    else {
      if (sourceTask == NULL)
	cerr << "ERROR: Undefined task, '" << source << "'" << endl;
      if (targetTask == NULL) 
	cerr << "ERROR: Undefined task, '" << target << "'" << endl;
    }
  }
}

GV_TaskGraph::~GV_TaskGraph()
{
  for (list<GV_Task*>::iterator iter = m_tasks.begin();
       iter != m_tasks.end(); iter++)
    delete *iter;
  for (list<Edge*>::iterator iter = m_edges.begin();
       iter != m_edges.end(); iter++)
    delete *iter;
}

void GV_TaskGraph::topologicallySortEdges()
{
  list<GV_Task*>::iterator iter;
  for( iter = m_tasks.begin(); iter != m_tasks.end(); iter++ ) {
    GV_Task* task = *iter;
    task->resetFlags();
  }

  vector<GV_Task*> sortedTasks;
  for( iter = m_tasks.begin(); iter != m_tasks.end(); iter++ ) {
    GV_Task* task = *iter;
    if(!task->sorted()){
      task->processTaskForSorting(sortedTasks);
    }
  }

  m_edges.clear();
  for (int i = 0; i < (int)sortedTasks.size(); i++) {
    list<Edge*> dependentEdges = sortedTasks[i]->getDependentEdges();
    for (list<Edge*>::iterator edgeIter = dependentEdges.begin();
	 edgeIter != dependentEdges.end(); edgeIter++) {
      m_edges.push_back(*edgeIter);
    }
  }
}

void
GV_Task::processTaskForSorting(vector<GV_Task*>& sortedTasks)
{
  if(m_visited){
    cerr << "Cycle detected in task graph: already did\n\t"
	 << getName() << endl;
    exit(1);
  }

  m_visited=true;
   
  list<Edge*>::iterator edgeIter;
  for (edgeIter = m_dependencyEdges.begin();
       edgeIter != m_dependencyEdges.end(); edgeIter++) {
    GV_Task* source = (*edgeIter)->getSource();
    if(!source->m_sorted){
      if(source->m_visited){
	cerr << "Cycle detected in task graph: trying to do\n\t"
	     << getName();
	cerr << "\nbut already did:\n\t"
	     << source->getName() << endl;
	exit(1);
      }
      source->processTaskForSorting(sortedTasks);
    }
  }

  // All prerequisites are done - add this task to the list
  sortedTasks.push_back(this);
  m_sorted=true;
}

void GV_TaskGraph::computeMaxPathLengths()
{
  topologicallySortEdges();
  markObsoleteEdges();
  
  list<GV_Task*>::iterator task_it;
  list<Edge*>::iterator it;
  list<Edge*>::reverse_iterator r_it;

  // sets the max_below_cost's
  for (r_it = m_edges.rbegin(); r_it != m_edges.rend(); r_it++)
    (*r_it)->relaxEdgeUp();

  // sets the max_above_costs's
  for (it = m_edges.begin(); it != m_edges.end(); it++) {
    (*it)->relaxEdgeDown();
  }

  // set critical path cost
  for (task_it = m_tasks.begin(); task_it != m_tasks.end(); task_it++) {
    if ((*task_it)->getMaxInclBelowCost() > m_criticalPathCost)
      m_criticalPathCost = (*task_it)->getMaxInclBelowCost();
  }

  cout << "Processed " << m_tasks.size() << " nodes and "
       << m_edges.size() << " edges" << endl;  
}

void GV_TaskGraph::markObsoleteEdges()
{
  // Mark all edges that are obsolete.  An obsolete edge is one that
  // is not the only path from its source to its target.
  
  // This map's source tasks to each possible destination with
  // a count of how many paths there are to each destination.
  map<GV_Task*, map<GV_Task*, int> > pathCountMap;

  // relies upon the edges being topologically sorted
  list<Edge*>::reverse_iterator r_it;
  map<GV_Task*, int>::iterator foundIt;
  for (r_it = m_edges.rbegin(); r_it != m_edges.rend(); r_it++) {
    Edge* edge = *r_it;
    map<GV_Task*, int>& destMap = pathCountMap[edge->getSource()];

    // add direct path to the path count
    foundIt = destMap.find(edge->getTarget());
    if (foundIt != destMap.end())
      (*foundIt).second++;
    else
      destMap[edge->getTarget()] = 1;

    // add indirect paths to path counts
    map<GV_Task*, int>& indirectDestMap = pathCountMap[edge->getTarget()];
    for (map<GV_Task*, int>::iterator destIter = indirectDestMap.begin();
	 destIter != indirectDestMap.end(); destIter++) {
      foundIt = destMap.find((*destIter).first);
      if (foundIt != destMap.end()) {
	(*foundIt).second += (*destIter).second;
      }
      else {
	destMap[(*destIter).first] = (*destIter).second;
      }
    }
  }

  for (r_it = m_edges.rbegin(); r_it != m_edges.rend(); r_it++) {
    Edge* edge = *r_it;
    int pathCount = pathCountMap[edge->getSource()][edge->getTarget()];
    assert(pathCount >= 1);
    if (pathCount > 1) {
      // There is more than one path from the source to the target, so
      // the direct path is obsolete when considering worrying about
      // critical paths.
      edge->setObsolete();
    }
  }
}

GV_Task*
GV_TaskGraph::findTask(string name)
{
  map<string, GV_Task*>::iterator iter = m_taskMap.find(name);
  if (iter == m_taskMap.end())
    return 0;
  return iter->second;
}

Edge*
GV_TaskGraph::findEdge(string name)
{
  map<string, Edge*>::iterator iter = m_edgeMap.find(name);
  if (iter == m_edgeMap.end())
    return 0;
  return iter->second;
}

