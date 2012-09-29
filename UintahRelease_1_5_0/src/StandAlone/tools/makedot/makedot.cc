/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/DataArchive/DataArchive.h>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
#include "DaVinci.h"
#include "graphview.h"
#include "GV_TaskGraph.h"

using namespace std;
using namespace Uintah;

bool load_timestep(int timestep, float prune_percent);

void usage(char* prog_name)
{
  cerr << "usage: " << prog_name
       << " <uda directory> [-t <timestep>] [-p <prune percent>] [-x]" << endl;
  cerr << endl << "Options\n";
  cerr << "-t <timestep>\n"
      << "\tLoads the taskgraph from the given timestep directory in the uda\n"
      << "\tdirectory.\n";
  cerr << "-p <prune percent>\n"
      << "\tHide nodes and edges with maximum path costs less than <percent>\n"
       << "\tof the critical path cost.\n";
  cerr << "-x\n"
    << "\tNot just hide, but exclude nodes with maximum path costs less than\n"
    << "\tthe set pruning percent.  This is useful for very large graphs.\n";
}

int
main(int argc, char* argv[])
{
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }

  udaDir = argv[1];

  int timestep = 1;
  float prune_percent = 0;
  bool do_exclusion = false;
  
  for (int i = 2; i < argc; i++) {
    if (argv[i][0] == '-') {
      if (argv[i][1] == 't') {
	if (++i >= argc) {
	  usage(argv[0]);
	  return 1;
	}
	timestep = atoi(argv[i]);
      }
      else if (argv[i][1] == 'p') {
	if (++i >= argc) {
	  usage(argv[0]);
	  return 1;
	}
	prune_percent = atof(argv[i]);
      }
      else if (argv[i][1] == 'x')
	do_exclusion = true;
      else {
	cerr << "Invalid option " << argv[i] << endl;
	usage(argv[0]);
	return 1;
      }
    }
    else {
      usage(argv[0]);
      return 1;
    }
  }
  
  bool loaded = load_timestep(timestep, prune_percent);
  if (!loaded) {
    cerr << "Failed reading task graph.  Quitting.\n";
    return 1;
  }
}

bool load_timestep(int timestep, float prune_percent)
{
  ostringstream timedir;
  timedir << "/t" << setw(5) << setfill('0') << timestep;
  cout << "Loading timestep " << timestep << "...\n";

  int process = 0;
  string xmlFileName;
  FILE* tstFile;
  map<string, int> taskNumbers;
  int nextTask = 0;
  ostringstream outname;
  outname << "graph_" << timestep << ".dot";
  ofstream out(outname.str().c_str()); 
  out << "digraph G {\n";
  do {
    ostringstream pname;
    pname << "/taskgraph_" << setw(5) << setfill('0') << process << ".xml";
    xmlFileName = xmlDir + pname.str();
    
    if ((tstFile = fopen(xmlFileName.c_str(), "r")) == NULL)
      break;
    fclose(tstFile);

    ProblemSpecReader psr(xmlFileName);

    ProblemSpecP prob_spec = psr.readInputFile();
 
    ProblemSpecP nodes = prob_spec->findBlock("Nodes");
    for (ProblemSpecP node = nodes->findBlock("node"); node != 0;
         node = node->findNextBlock("node")) {
      string task_name;
      double task_duration;
      double exec_duration;
      node->get("name", task_name);
      node->get("duration", task_duration);
      node->get("execution_duration", exec_duration);
    
      taskNumbers[task_name] = nextTask++;

      int num = taskNumbers[task_name];

      out << "  n" << num << " [shape=circle,label=\"" << task_name << ", time=" << exec_duration << ", " << task_duration << ", " << exec_percent << "\"];\n";
    }

    ProblemSpecP edges = prob_spec->findBlock("Edges");
    for (ProblemSpecP node = edges->findBlock("edge"); node != 0;
         node = node->findNextBlock("edge")) {
      string source;
      string target;
      node->get("source", source);
      node->get("target", target);
      int sourcenode = taskNumbers[source];
      int targetnode = taskNumbers[target];

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
    process++;
  } while (process < 100000 /* it will most likely always break out of loop
			       -- but just so it won't ever be caught in an
			       infinite loop */);  
  if (process == 0) {
    cerr << "Task graph data does not exist:" << endl;
    cerr << xmlFileName << " does not exist." << endl;
    return false;
  }
  
  return true;
}
