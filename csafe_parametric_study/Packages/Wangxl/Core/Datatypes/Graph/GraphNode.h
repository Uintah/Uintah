#ifndef SCI_Wangxl_Datatypes_GraphNode_h
#define SCI_Wangxl_Datatypes_GraphNode_h

#include "GraphEdge.h"

#include <list>
#include <iostream>

namespace Wangxl {

using std::list;
using std::iostream;

class NodeData;

typedef std::iterator<std::bidirectional_iterator_tag, GraphEdge> bi_iter_edge;
typedef std::iterator<std::bidirectional_iterator_tag, GraphNode> bi_iter_node;

class GraphNode {
public:
  GraphNode();
  //  int getDegree() const;
  int id() const;
  const GraphNode& opposite(GraphEdge edge) const;

  typedef list<GraphEdge>::const_iterator edges_iterator;

  class adj_edges_iterator : bi_iter_edge {
  public:
    adj_edges_iterator();
    adj_edges_iterator(GraphNode, bool);
    
    bool operator==(const adj_edges_iterator&) const;
    bool operator!=(const adj_edges_iterator&) const;
    
    adj_edges_iterator& operator++();
    adj_edges_iterator operator++(int);
    adj_edges_iterator& operator--();
    adj_edges_iterator operator--(int);
    
    const GraphEdge& operator*() const;
    const GraphEdge* operator->() const;
  private:
    edges_iterator akt_edge, last_edge, begin_edge;
  };
  
  class adj_nodes_iterator : bi_iter_node {
  public:
    
    // constructor
    adj_nodes_iterator();
    adj_nodes_iterator(const GraphNode&, bool);
    
    // comparibility
    bool operator==(const adj_nodes_iterator&) const;
    bool operator!=(const adj_nodes_iterator&) const;
    
    // operators
    adj_nodes_iterator& operator++();
    adj_nodes_iterator operator++(int);
    adj_nodes_iterator& operator--();
    adj_nodes_iterator operator--(int);

    // dereferencing
    const GraphNode& operator*() const;
    const GraphNode* operator->() const;
    
  private:
    adj_edges_iterator akt_edge;
    const GraphNode *int_node;
  };
  
  adj_nodes_iterator adj_nodes_begin() const;
  adj_nodes_iterator adj_nodes_end() const;
  adj_edges_iterator adj_edges_begin() const;
  adj_edges_iterator adj_edges_end() const; 
  edges_iterator edges_begin() const;
  edges_iterator edges_end() const;
private:
  NodeData* data;
  
  friend class Graph;
  friend class GraphEdge;
  friend class adj_nodes_iterator;
  
  friend bool operator==(GraphNode, GraphNode);
  friend bool operator!=(GraphNode, GraphNode);
  friend bool operator<(GraphNode, GraphNode);
  friend ostream& operator << (ostream& os, const GraphNode& node);
  
};

}

#endif
