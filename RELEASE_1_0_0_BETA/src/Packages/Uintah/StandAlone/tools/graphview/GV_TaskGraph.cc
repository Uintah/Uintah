#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <Dataflow/XMLUtil/SimpleErrorHandler.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include "GV_TaskGraph.h"
#include <Core/Malloc/Allocator.h>

using namespace std;
using namespace SCIRun;

float safePercent(double num, double denom)
{ return (denom != 0) ? num / denom: 0; }

GV_Task::GV_Task(string name, double duration, GV_TaskGraph* owner)
  : m_name(name), m_duration(duration),
    m_graph(owner),
    m_maxBelowCost(0), m_maxAboveCost(0)
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
    if (task == (*iter)->getTarget())
      return 0;
  }

  Edge* newEdge = scinew Edge(this, task);
  
  m_dependencyEdges.push_back(newEdge);
  task->m_dependentEdges.push_back(newEdge); 
  return newEdge;
}

Edge::Edge(GV_Task* dependent, GV_Task* dependency)
  : m_source(dependent), m_target(dependency)
{}

float Edge::getMaxPathPercent() const
{
  return safePercent(getMaxInclusivePathCost(),
		     m_target->m_graph->getCriticalPathCost());
}

void Edge::relaxEdgeUp()
{
  m_target->testSetMaxBelowCost(m_source->getMaxInclBelowCost());
}

void Edge::relaxEdgeDown()
{
  m_source->testSetMaxAboveCost(m_target->getMaxInclAboveCost());
} 


GV_TaskGraph*
GV_TaskGraph::inflate(string xmlDir)
{
  try {
    XMLPlatformUtils::Initialize();
  } catch (const XMLException& e) {
    cerr << "Unable to initialize XML library: " << e.getMessage() << endl;
    return 0;
  }

  DOM_Document firstDoc;

  GV_TaskGraph* pGraph = scinew GV_TaskGraph();
  
  DOMParser parser;
  parser.setDoValidation(false);
  SimpleErrorHandler handler;
  parser.setErrorHandler(&handler);

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

    parser.parse(xmlFileName.c_str());
    if (handler.foundError) {
      cerr << "Error parsing taskgraph file " << xmlFileName << endl;
      return 0;
    }

    if (process == 0)
      firstDoc = parser.getDocument();

    pGraph->readNodes(parser.getDocument());
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
  
  // get the edges from the last document
  // (they should all contain the same edges)
  pGraph->readEdges(firstDoc);
  
  return pGraph;
}

GV_TaskGraph::GV_TaskGraph()
  : m_criticalPathCost(0), m_thresholdPercent(0)
{
}

void GV_TaskGraph::readNodes(DOM_Document xmlDoc)
{
  DOM_Element docElement = xmlDoc.getDocumentElement();
  DOM_Node nodes = findNode("Nodes", docElement);
  for (DOM_Node node = findNode("node", nodes); node != 0;
       node = findNextNode("node", node)) {
    string task_name;
    double task_duration;
    get(node, "name", task_name);
    get(node, "duration", task_duration);
    
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

void GV_TaskGraph::readEdges(DOM_Document xmlDoc)
{
  DOM_Element docElement = xmlDoc.getDocumentElement();
  DOM_Node edges = findNode("Edges", docElement);
  for (DOM_Node node = findNode("edge", edges); node != 0;
       node = findNextNode("edge", node)) {
    string source;
    string target;
    get(node, "source", source);
    get(node, "target", target);
    GV_Task* sourceTask = m_taskMap[source];
    GV_Task* targetTask = m_taskMap[target];

    if (sourceTask != NULL && targetTask != NULL) {
      Edge* edge = sourceTask->addDependency(m_taskMap[target]);
      if (edge) {
	m_edges.push_back(edge);
	m_edgeMap[source + "->" + target] = edge;
      }
    }
    else {
      if (sourceTask == NULL)
	cerr << "ERROR: Undefined task, '" << source << "'" << endl;
      if (targetTask == NULL) 
	cerr << "ERROR: Undefined task, '" << target << "'" << endl;
    }
  }
  cout << "Processed " << m_tasks.size() << " nodes and "
       << m_edges.size() << " edges" << endl;

  computeMaxPathLengths();
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

void GV_TaskGraph::computeMaxPathLengths()
{
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

