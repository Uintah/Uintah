#ifndef SCI_Wangxl_Datatypes_GraphEdge_h
#define SCI_Wangxl_Datatypes_GraphEdge_h

#include <list>
#include <iostream>

namespace Wangxl {

using std::list;
using std::ostream;

class GraphNode;
class EdgeData;

class GraphEdge {
public:
  GraphEdge();
  int id();
  bool isHidden();
  GraphNode source() const;
  GraphNode target() const;
private:
  EdgeData *data;
  
  friend class Graph;
  friend class GraphNode;
  
  friend bool operator==(GraphEdge, GraphEdge);
  friend bool operator!=(GraphEdge, GraphEdge);
  friend bool operator<(GraphEdge, GraphEdge);
  friend ostream& operator<< (ostream& os, const GraphEdge& edge);
};

}

#endif
