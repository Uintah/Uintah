
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Core/Exceptions/ErrnoException.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/XMLUtil/XMLUtil.h>

#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <errno.h>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <time.h>

using namespace Uintah;
using namespace SCIRun;

using std::cerr;
using std::string;

Scheduler::Scheduler(Output* oport)
  : m_outPort(oport), m_graphDoc(NULL), m_nodes(NULL),
    m_doEmitTaskGraphDocs(false) //, m_executeCount(0)
{
}

Scheduler::~Scheduler()
{
}

void
Scheduler::makeTaskGraphDoc(const vector<Task*>& tasks, bool emit_edges /* = true */)
{
  if (!m_doEmitTaskGraphDocs)
    return;
  
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


