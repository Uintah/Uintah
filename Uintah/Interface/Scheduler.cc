/* REFERENCED */
static char *id="@(#) $Id$";

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <Uintah/Components/Schedulers/TaskGraph.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include "Scheduler.h"
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah;
using namespace PSECore::XMLUtil;
using namespace std;

Scheduler::Scheduler(Output* oport)
  : m_outPort(oport), m_graphDoc(NULL), m_nodes(NULL), m_executeCount(0)
{
}

Scheduler::~Scheduler()
{
}

void
Scheduler::emitEdges(const vector<Task*>& tasks)
{
    // We'll get called once for each timestep. Each of these executions should
    // have the same task graph, execept for the output task which only happens
    // periodically. The first execution will have the output task though, so
    // we'll just do that. The only complication is that the very first call is
    // actually just some initialization tasks, so we really want to output the
    // second call (which is for the first timestep).
    if (m_executeCount++ != 1)
    	return;
    
    DOM_DOMImplementation impl;
    m_graphDoc = scinew DOM_Document();
    *m_graphDoc = impl.createDocument(0, "Uintah_TaskGraph", DOM_DocumentType());
    DOM_Element root = m_graphDoc->getDocumentElement();
    
    DOM_Element meta = m_graphDoc->createElement("Meta");
    root.appendChild(meta);
    appendElement(meta, "username", getenv("LOGNAME"));
    time_t t = time(NULL);
    appendElement(meta, "date", ctime(&t));

    DOM_Element edges = m_graphDoc->createElement("Edges");
    root.appendChild(edges);

    m_nodes = scinew DOM_Element(m_graphDoc->createElement("Nodes"));
    root.appendChild(*m_nodes);

    // Now that we've build the XML structure, we can add the actual edges
    map<TaskProduct, Task*> computes_map;
    vector<Task*>::const_iterator task_iter;
    for (task_iter = tasks.begin(); task_iter != tasks.end(); task_iter++) {
    	const vector<Task::Dependency*>& comps = (*task_iter)->getComputes();
    	for (vector<Task::Dependency*>::const_iterator dep_iter = comps.begin();
	     dep_iter != comps.end(); dep_iter++) {
    	    const Task::Dependency* dep = *dep_iter;
	    TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
	    computes_map[p] = *task_iter;
	}
    }

    string depfile_name(m_outPort->getOutputLocation() + "/taskgraph");
    ofstream depfile(depfile_name.c_str());
    if (!depfile) {
	cerr << "Scheduler::emitEdges(): unable to open output file!\n";
	return;	// dependency dump failure shouldn't be fatal to anything else
    }

    for (task_iter = tasks.begin(); task_iter != tasks.end(); task_iter++) {
    	const Task* task = *task_iter;

	const vector<Task::Dependency*>& deps = task->getRequires();
	vector<Task::Dependency*>::const_iterator dep_iter;
	for (dep_iter = deps.begin(); dep_iter != deps.end(); dep_iter++) {
	    const Task::Dependency* dep = *dep_iter;

	    if (!dep->d_dw->isFinalized()) {

		TaskProduct p(dep->d_patch, dep->d_matlIndex, dep->d_var);
		map<TaskProduct, Task*>::const_iterator deptask =
	    	    computes_map.find(p);

		const Task* task1 = task;
		const Task* task2 = deptask->second;

    	    	ostringstream name1;
		name1 << task1->getName();
		if (task1->getPatch())
		    name1 << "\\nPatch" << task1->getPatch()->getID();
		
		ostringstream name2;
		name2 << task2->getName();
		if (task2->getPatch())
		    name2 << "\\nPatch" << task2->getPatch()->getID();
		    
    	    	depfile << "\"" << name1.str() << "\" \""
		    	<< name2.str() << "\"\n";

    	    	DOM_Element edge = edges.getOwnerDocument().createElement("edge");
    	    	appendElement(edge, "source", name1.str());
		appendElement(edge, "target", name2.str());
    	    	edges.appendChild(edge);
	    }
	}
    }

    depfile.close();
}

void
Scheduler::emitNode(const Task* task, time_t start, double duration)
{
    if (m_nodes == NULL)
    	return;
    
    DOM_Element node = m_graphDoc->createElement("node");
    m_nodes->appendChild(node);
    
    ostringstream name;
    name << task->getName();
    if (task->getPatch())
    	name << "\\nPatch" << task->getPatch()->getID();
    appendElement(node, "name", name.str());
    appendElement(node, "start", ctime(&start));
    appendElement(node, "duration", duration);
}

void
Scheduler::finalizeNodes()
{
    if (m_graphDoc == NULL)
    	return;
    
    string deps_name(m_outPort->getOutputLocation() + "/taskgraph.xml");
    ofstream deps(deps_name.c_str());
    deps << *m_graphDoc << endl;
    
    delete m_nodes;
    m_nodes = NULL;
    delete m_graphDoc;
    m_graphDoc = NULL;
}

//
// $Log$
// Revision 1.8  2000/08/08 01:32:48  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.7  2000/07/26 20:14:13  jehall
// Moved taskgraph/dependency output files to UDA directory
// - Added output port parameter to schedulers
// - Added getOutputLocation() to Uintah::Output interface
// - Renamed output files to taskgraph[.xml]
//
// Revision 1.6  2000/07/25 20:59:27  jehall
// - Simplified taskgraph output implementation
// - Sort taskgraph edges; makes critical path algorithm eastier
//
// Revision 1.5  2000/07/25 17:55:27  jehall
// - Added include in case the MIPSPro CC decides to use this file to
//   instantiate Handle<DataWarehouse>.
//
// Revision 1.4  2000/07/19 21:41:53  jehall
// - Added functions for emitting task graph information to reduce redundancy
//
// Revision 1.3  2000/04/26 06:49:12  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/03/16 22:08:23  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
