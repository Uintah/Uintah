//
// $Id$
//

#include <Uintah/Interface/Scheduler.h>

#include <SCICore/Exceptions/ErrnoException.h>
#include <SCICore/Malloc/Allocator.h>
#include <PSECore/XMLUtil/XMLUtil.h>

#include <Uintah/Components/Schedulers/TaskGraph.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Interface/DataWarehouse.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <errno.h>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <time.h>

using namespace Uintah;
using namespace PSECore::XMLUtil;

using std::cerr;
using std::string;
using namespace SCICore::OS;
using namespace SCICore::Exceptions;

Scheduler::Scheduler(Output* oport)
  : m_outPort(oport), m_graphDoc(NULL), m_nodes(NULL)//, m_executeCount(0)
{
}

Scheduler::~Scheduler()
{
}

void
Scheduler::makeTaskGraphDoc(const vector<Task*>& tasks, bool emit_edges /* = true */)
{
    if (!m_outPort->wasOutputTimestep())
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

    m_nodes = scinew DOM_Element(m_graphDoc->createElement("Nodes"));
    root.appendChild(*m_nodes);

    if (!emit_edges)
      return;
    
    DOM_Element edges = m_graphDoc->createElement("Edges");
    root.appendChild(edges);

    // Now that we've build the XML structure, we can add the actual edges
    map<TaskProduct, Task*> computes_map;
    vector<Task*>::const_iterator task_iter;
    for (task_iter = tasks.begin(); task_iter != tasks.end(); task_iter++) {
    	const Task::compType& comps = (*task_iter)->getComputes();
    	for (Task::compType::const_iterator dep = comps.begin();
	     dep != comps.end(); dep++) {
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

	const Task::reqType& deps = task->getRequires();
	Task::reqType::const_iterator dep;
	for (dep = deps.begin(); dep != deps.end(); dep++) {
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
Scheduler::finalizeNodes(int process /* = 0*/)
{
    if (m_graphDoc == NULL)
    	return;

    if (m_outPort->wasOutputTimestep()) {
      string timestep_dir(m_outPort->getLastTimestepOutputLocation());
      
      ostringstream fname;
      fname << "/taskgraph_" << setw(5) << setfill('0') << process << ".xml";
      string file_name(timestep_dir + fname.str());
      ofstream graphfile(file_name.c_str());
      graphfile << *m_graphDoc << endl;
    }
    
    delete m_nodes;
    m_nodes = NULL;
    delete m_graphDoc;
    m_graphDoc = NULL;
}

void
Scheduler::problemSetup(const ProblemSpecP&)
{
   // For schedulers that need no setup
}


//
// $Log$
// Revision 1.15  2000/12/10 09:06:21  sparker
// Merge from csafe_risky1
//
// Revision 1.14.2.1  2000/10/10 05:28:10  sparker
// Added support for NullScheduler (used for profiling taskgraph overhead)
//
// Revision 1.14  2000/09/29 05:35:07  sparker
// Quiet g++ warnings
//
// Revision 1.13  2000/09/27 00:14:33  witzel
// Changed emitEdges to makeTaskGraphDoc with an option to emit
// the actual edges (only process 0 in the MPI version since all
// process contain the same taskgraph edge information).
//
// Revision 1.12  2000/09/26 21:41:38  dav
// minor formatting/include rearrangment
//
// Revision 1.11  2000/09/25 20:39:14  sparker
// Quiet g++ compiler warnings
//
// Revision 1.10  2000/09/20 15:50:30  sparker
// Added problemSetup interface to scheduler
// Added ability to get/release the loadBalancer from the scheduler
//   (used for getting processor assignments to create per-processor
//    tasks in arches)
// Added getPatchwiseProcessorAssignment to LoadBalancer interface
//
// Revision 1.9  2000/09/08 17:49:50  witzel
// Changing finalizeNodes so that it outputs different taskgraphs
// in different timestep directories and the taskgraph information
// of different processes in different files.
//
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
