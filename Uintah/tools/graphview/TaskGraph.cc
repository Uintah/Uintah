#include <iostream>
#include <PSECore/XMLUtil/SimpleErrorHandler.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include "TaskGraph.h"

using namespace std;
using namespace PSECore::XMLUtil;

Task::Task(string name, double duration)
  : m_name(name), m_duration(duration)
{}

Task::~Task()
{}

Edge*
Task::addDependency(Task* task)
{
    for (list<Task*>::const_iterator iter = m_dependencies.begin();
    	 iter != m_dependencies.end(); iter++) {
    	if (task == *iter)
	    return 0;
    }

    m_dependencies.push_back(task);
    task->m_dependents.push_back(this);

    return new Edge(this, task);
}

Edge::Edge(Task* dependent, Task* dependency)
  : m_source(dependent), m_target(dependency)
{}

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
{
    DOM_Element docElement = xmlDoc.getDocumentElement();

    DOM_Node nodes = findNode("Nodes", docElement);
    for (DOM_Node node = findNode("node", nodes); node != 0;
    	 node = findNextNode("node", node)) {
    	string task_name;
	double task_duration;
	get(node, "name", task_name);
    	get(node, "duration", task_duration);

    	Task* task = new Task(task_name, task_duration);
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
    	if (edge)
    	    m_edges.push_back(edge);
    }
    cout << "Processed " << m_tasks.size() << " nodes and "
    	 << m_edges.size() << " edges" << endl;
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

Task*
TaskGraph::find(string name)
{
    map<string, Task*>::iterator iter = m_taskMap.find(name);
    if (iter == m_taskMap.end())
    	return 0;
    return iter->second;
}
