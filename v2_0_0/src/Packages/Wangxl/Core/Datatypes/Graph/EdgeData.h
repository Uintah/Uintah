#ifndef SCI_Wangxl_Datatypes_EdgeData_h
#define SCI_Wangxl_Datatypes_EdgeData_h

#include "GraphNode.h"
#include "GraphEdge.h"

#include <list>

namespace Wangxl {

using std::list;

class Graph;

class EdgeData {
public:
  int id;				// internal numbering
  Graph* owner;
  GraphNode nodes[2]; 		// nodes[0] = sources,nodes[1] = targets
  list<GraphEdge>::iterator adj_pos[2];// positions in the adjacency lists
                                   // of sources and targets
  list<GraphEdge>::iterator pos;		// position in the list of all edges
  bool hidden;
};

}

#endif
