#include "Graph.h"
#include "NodeData.h"
#include "EdgeData.h"

namespace Wangxl {

Graph::Graph() : ndnum(0), ednum(0), hndnum(0), hednum(0), fnidnum(0), feidnum(0) {}
Graph::~Graph()
{
  //clear();
}

GraphNode Graph::newNode()
{
GraphNode node;

  node.data = new NodeData;
  node.data->id = newNodeId();
  node.data->owner = this;
  node.data->pos = nodes.insert(nodes.end(), node);
  node.data->hidden = false;
  ++ndnum;
  return node;
}

GraphEdge Graph::newEdge(GraphNode source, GraphNode target)
{
GraphEdge edge;

  assert(source.data);
  assert(target.data);
  assert(source.data->owner == this);
  assert(target.data->owner == this);

  edge.data = new EdgeData;
  edge.data->owner = this;
  edge.data->id = newEdgeId();	
  edge.data->nodes[0] = source;
  edge.data->nodes[1] = target;
  edge.data->pos = edges.insert(edges.end(), edge);
  edge.data->hidden = false;
  ++ednum;

  list<GraphEdge> &source_adj = source.data->edges;
  list<GraphEdge> &target_adj = target.data->edges;
  
  edge.data->adj_pos[0] = source_adj.insert(source_adj.begin(), edge);
  edge.data->adj_pos[1] = target_adj.insert(target_adj.begin(), edge);
  
  return edge;
}

int Graph::newNodeId()
{
  if(fnids.empty()) return ndnum;
  int id = fnids.back();
  fnids.pop_back();
  --fnidnum;
  return id;
}

int Graph::newEdgeId()
{
  if(feids.empty()) return ednum;
  int id = feids.back();
  feids.pop_back();
  --feidnum;
  return id;
}

void Graph::hideEdge (GraphEdge edge) 
{
    assert (edge.data->owner == this);
    if (!edge.isHidden()) {
      // remove from ajacent nodes' edges list
      edge.data->nodes[0].data->edges.erase( edge.data->adj_pos[0] );
      edge.data->nodes[1].data->edges.erase( edge.data->adj_pos[1] );

      //remove from edges list of the graph
      edges.erase (edge.data->pos);

      //
      edge.data->pos = hedges.insert(hedges.end(), edge);
      edge.data->hidden = true;
      ++hednum;
    }
}


int Graph::getNodesNum() const
{
  return ndnum - hndnum;
}
int Graph::getEdgesNum() const
{
  return ednum - hednum;
}

int Graph::getIdsNum( GraphNode ) const
{
  return fnidnum + ndnum;
}

int Graph::getIdsNum( GraphEdge ) const
{
  return feidnum + ednum;
}

Graph::node_iterator Graph::nodes_begin() const
{
    return nodes.begin();
}

Graph::node_iterator Graph::nodes_end() const
{
    return nodes.end();
}

Graph::edge_iterator Graph::edges_begin() const
{
    return edges.begin();
}

Graph::edge_iterator Graph::edges_end() const
{
    return edges.end();
}

}
