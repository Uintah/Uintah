#include "NodeData.h"
#include "EdgeData.h"
#include "Graph.h"

#include <iostream>

namespace Wangxl {

using std::ostream;

GraphNode::GraphNode() : data(0) {}

ostream& operator << (ostream& os, const GraphNode& node) {
    return os << "[" << node.id() << "]";
}

bool operator == ( GraphNode node0, GraphNode node1 )
{
  return node0.data == node1.data;
}

bool operator != ( GraphNode node0, GraphNode node1 )
{
  return node0.data != node1.data;
}

bool operator < ( GraphNode node0, GraphNode node1 )
{
  return node0.data < node1.data;
}
GraphNode::edges_iterator GraphNode::edges_begin() const
{
    return data->edges.begin();
}

GraphNode::edges_iterator GraphNode::edges_end() const
{
    return data->edges.end();
}

GraphNode::adj_nodes_iterator GraphNode::adj_nodes_begin() const
{ 
    return GraphNode::adj_nodes_iterator(*this, true);
}

GraphNode::adj_nodes_iterator GraphNode::adj_nodes_end() const
{
    return GraphNode::adj_nodes_iterator(*this, false);
}

GraphNode::adj_edges_iterator GraphNode::adj_edges_begin() const
{
    return GraphNode::adj_edges_iterator(*this, true);
}

GraphNode::adj_edges_iterator GraphNode::adj_edges_end() const
{
    return GraphNode::adj_edges_iterator(*this, false);
}

int GraphNode::id() const
{
    return data->id;
}

const GraphNode& GraphNode::opposite(GraphEdge edge) const
{
    assert(edge.data);

    GraphNode& s = edge.data->nodes[0];
    if (*this == s)
	return edge.data->nodes[1];
    else
	return s;
}

GraphNode::adj_edges_iterator::adj_edges_iterator()
{
}

GraphNode::adj_edges_iterator::adj_edges_iterator(GraphNode node, bool start)
{
    // iterators that are used everytime
  begin_edge = node.edges_begin();
  last_edge = node.edges_end();
  if ( start ) akt_edge = node.edges_begin();
  else akt_edge = node.edges_end();
}

bool GraphNode::adj_edges_iterator::operator==(const GraphNode::adj_edges_iterator& i) const
{
  return i.akt_edge == akt_edge;
}

bool GraphNode::adj_edges_iterator::operator!=(const GraphNode::adj_edges_iterator& i) const
{
  return i.akt_edge != akt_edge;
}

GraphNode::adj_edges_iterator& GraphNode::adj_edges_iterator::operator++()
{
  ++akt_edge;
  return *this;
}

GraphNode::adj_edges_iterator GraphNode::adj_edges_iterator::operator++(int)
{
    GraphNode::adj_edges_iterator tmp = *this;
    operator++();
    return tmp;
}

GraphNode::adj_edges_iterator& GraphNode::adj_edges_iterator::operator--()
{
  --akt_edge;
  return *this;
}

GraphNode::adj_edges_iterator GraphNode::adj_edges_iterator::operator--(int)
{
    GraphNode::adj_edges_iterator tmp = *this;
    operator--();
    return tmp;
}

const GraphEdge& GraphNode::adj_edges_iterator::operator*() const
{
    return *akt_edge;
}

const GraphEdge* GraphNode::adj_edges_iterator::operator->() const
{
    return akt_edge.operator->();
}

GraphNode::adj_nodes_iterator::adj_nodes_iterator()
{
}

GraphNode::adj_nodes_iterator::adj_nodes_iterator(const GraphNode& node, bool start)
{
  int_node = &node;
  if (start)
    akt_edge = node.adj_edges_begin();
  else
    akt_edge = node.adj_edges_end();
}

bool GraphNode::adj_nodes_iterator::operator==(const GraphNode::adj_nodes_iterator& i) const
{
  return i.akt_edge == akt_edge;
}

bool GraphNode::adj_nodes_iterator::operator!=(const GraphNode::adj_nodes_iterator& i) const
{
  return i.akt_edge != akt_edge;
}

GraphNode::adj_nodes_iterator& GraphNode::adj_nodes_iterator::operator++()
{
  ++akt_edge;
  return *this;
}

GraphNode::adj_nodes_iterator GraphNode::adj_nodes_iterator::operator++(int)
{
    GraphNode::adj_nodes_iterator tmp = *this;
    operator++();
    return tmp;
}

GraphNode::adj_nodes_iterator& GraphNode::adj_nodes_iterator::operator--()
{
    --akt_edge;
    return *this;
}

GraphNode::adj_nodes_iterator GraphNode::adj_nodes_iterator::operator--(int)
{
    GraphNode::adj_nodes_iterator tmp = *this;
    operator--();
    return tmp;
}

const GraphNode& GraphNode::adj_nodes_iterator::operator*() const
{
    return int_node->opposite(*akt_edge);
}

const GraphNode* GraphNode::adj_nodes_iterator::operator->() const
{
    return &int_node->opposite(*akt_edge);
}

}









