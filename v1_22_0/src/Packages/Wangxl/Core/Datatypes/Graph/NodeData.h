#ifndef SCI_Wangxl_Datatypes_NodeData_h
#define SCI_Wangxl_Datatypes_NodeData_h

#include "GraphNode.h"
#include "GraphEdge.h"

#include <list>

namespace Wangxl {

using std::list;

class Graph;

class NodeData {
public:
  int id;			// internal id number
  Graph *owner;		// graph containing this node
  list<GraphNode>::iterator pos;	// position in the list of all nodes
  list<GraphEdge> edges;	// edges incident to this node
  bool hidden; // flag indicating if it appears
};

}

#endif
