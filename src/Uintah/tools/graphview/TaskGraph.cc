#include <iostream>
#include <PSECore/XMLUtil/SimpleErrorHandler.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include "TaskGraph.h"

using namespace std;
using namespace PSECore::XMLUtil;

float safePercent(double num, double denom)
{ return (denom != 0) ? num / denom: 0; }

Task::Task(string name, double duration, TaskGraph* owner)
  : m_name(name), m_duration(duration),
    m_graph(owner),
    m_maxBelowCost(0), m_maxAboveCost(0)
{}

Task::~Task()
{}

float Task::getMaxPathPercent() const
{
  return safePercent(getMaxInclusivePathCost(),
		     m_graph->getCriticalPathCost());
}

Edge* Task::addDependency(Task* task)
{
  for (list<Edge*>::const_iterator iter = m_dependencyEdges.begin();
       iter != m_dependencyEdges.end(); iter++) {
    if (task == (*iter)->getTarget())
      return 0;
  }

  Edge* newEdge = new Edge(this, task);
  
  m_dependencyEdges.push_back(newEdge);
  task->m_dependentEdges.push_back(newEdge); 
  return newEdge;
}

Edge::Edge(Task* dependent, Task* dependency)
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


TaskGraph*
TaskGraph::inflate(string xmlFileName)
{
  try {
    XMLPlatformUtils::Initialize();
  } catch (const XMLException& e) {
    cerr << "Unable to initialize XML library: " << e.getMessage() << endl;
    return 0;
  }

  DOMParser parser;
  parser.setDoValidation(false);
  SimpleErrorHandler handler;
  parser.setErrorHandler(&handler);
  
  parser.parse(xmlFileName.c_str());
  if (handler.foundError) {
    cerr << "Error parsing taskgraph file " << xmlFileName << endl;
    return 0;
  }
  
  return new TaskGraph(parser.getDocument());
}

TaskGraph::TaskGraph(DOM_Document xmlDoc)
  : m_criticalPathCost(0), m_thresholdPercent(0)
{
  DOM_Element docElement = xmlDoc.getDocumentElement();
  
  DOM_Node nodes = findNode("Nodes", docElement);
  for (DOM_Node node = findNode("node", nodes); node != 0;
       node = findNextNode("node", node)) {
    string task_name;
    double task_duration;
    get(node, "name", task_name);
    get(node, "duration", task_duration);
    
    Task* task = new Task(task_name, task_duration, this);
    m_tasks.push_back(task);
    m_taskMap[task_name] = task;
  }
  
  DOM_Node edges = findNode("Edges", docElement);
  for (DOM_Node node = findNode("edge", edges); node != 0;
       node = findNextNode("edge", node)) {
    string source;
    string target;
    get(node, "source", source);
    get(node, "target", target);
    
    Edge* edge = m_taskMap[source]->addDependency(m_taskMap[target]);
    if (edge) {
      m_edges.push_back(edge);
      m_edgeMap[source + "->" + target] = edge;
    }
  }
  cout << "Processed " << m_tasks.size() << " nodes and "
       << m_edges.size() << " edges" << endl;

  computeMaxPathLengths();
}

TaskGraph::~TaskGraph()
{
  for (list<Task*>::iterator iter = m_tasks.begin();
       iter != m_tasks.end(); iter++)
    delete *iter;
  for (list<Edge*>::iterator iter = m_edges.begin();
       iter != m_edges.end(); iter++)
    delete *iter;
}

void TaskGraph::computeMaxPathLengths()
{
  list<Task*>::iterator task_it;
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

Task*
TaskGraph::findTask(string name)
{
  map<string, Task*>::iterator iter = m_taskMap.find(name);
  if (iter == m_taskMap.end())
    return 0;
  return iter->second;
}

Edge*
TaskGraph::findEdge(string name)
{
  map<string, Edge*>::iterator iter = m_edgeMap.find(name);
  if (iter == m_edgeMap.end())
    return 0;
  return iter->second;
}

