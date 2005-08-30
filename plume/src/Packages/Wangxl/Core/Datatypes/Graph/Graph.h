#ifndef SCI_Wangxl_Datatypes_Graph_h
#define SCI_Wangxl_Datatypes_Graph_h

#include "GraphNode.h"
#include "GraphEdge.h"

namespace Wangxl {

class Graph {
public:
  Graph();
  //  Graph( const Graph& graph );
  //  Graph( const Graph& graph, const list< GraphNode >& nodes );
  virtual ~Graph();

  int getNodesNum() const;
  int getEdgesNum() const;
  int getIdsNum( GraphNode ) const;
  int getIdsNum( GraphEdge ) const;

  virtual GraphNode newNode();
  virtual GraphEdge newEdge(GraphNode s, GraphNode t);

  int newNodeId();
  int newEdgeId();
  //  void rmNode( GraphNode node );
  //  void rmEdge( GraphEdge edge );

  typedef list<GraphNode>::const_iterator node_iterator;
  typedef list<GraphEdge>::const_iterator edge_iterator;

  node_iterator nodes_begin() const;
  node_iterator nodes_end() const;
  edge_iterator edges_begin() const;
  edge_iterator edges_end() const;

  void hideEdge (GraphEdge edge);
  //  void restore_edge (GraphEdge edge);

private:
  list<GraphNode> nodes; // node list
  list<GraphEdge> edges; // edge list
  int ndnum, ednum; // node, edge numbers

  list<GraphNode> hnodes;
  list<GraphEdge> hedges;
  int hndnum, hednum; // hidden node, edge numbers

  list<int> fnids; // free node ids
  list<int> feids; // free edge ids
  int fnidnum, feidnum; // free node, edge numbers

};

}

#endif
