/* REFERENCED */
static char *id="@(#) $Id$";

#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <Uintah/Components/Schedulers/TaskGraph.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include "Scheduler.h"

using namespace Uintah;
using namespace PSECore::XMLUtil;
using namespace std;

Scheduler::Scheduler()
  : m_graphDoc(NULL), m_nodes(NULL), m_executeCount(0)
{
}

Scheduler::~Scheduler()
{
}

void
Scheduler::emitEdges(const TaskGraph& graph)
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
    m_graphDoc = new DOM_Document();
    *m_graphDoc = impl.createDocument(0, "Uintah_TaskGraph", DOM_DocumentType());
    DOM_Element root = m_graphDoc->getDocumentElement();
    
    DOM_Element meta = m_graphDoc->createElement("Meta");
    root.appendChild(meta);
    appendElement(meta, "username", getenv("LOGNAME"));
    time_t t = time(NULL);
    appendElement(meta, "date", ctime(&t));

    DOM_Element edges = m_graphDoc->createElement("Edges");
    root.appendChild(edges);
    graph.dumpDependencies(edges);

    m_nodes = new DOM_Element(m_graphDoc->createElement("Nodes"));
    root.appendChild(*m_nodes);
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
    
    ofstream deps("dependencies.xml");
    deps << *m_graphDoc << endl;
    
    delete m_nodes;
    m_nodes = NULL;
    delete m_graphDoc;
    m_graphDoc = NULL;
}

//
// $Log$
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
