#ifndef SCI_Wangxl_Datatypes_Dfs_h
#define SCI_Wangxl_Datatypes_Dfs_h

#include "Graph.h"
#include "NodeMap.h"

namespace Wangxl {

class Dfs {
public:
  Dfs();
  virtual ~Dfs();
  void run( Graph& graph );
  void start( const GraphNode& node ) { d_start = node; }
  GraphNode start() const { return d_start; }
  virtual void pre_recusive_handler( Graph& graph, GraphEdge& edge, GraphNode& node ) {}
private:
  void dfs( Graph& graph, GraphNode& curr, GraphNode& father );
protected:
  NodeMap<int> d_visit ;
  int d_dfsnum;
  GraphNode d_start;
};

Dfs::Dfs()
{
  d_dfsnum = 1;
}

Dfs::~Dfs()
{

}

void Dfs::run( Graph& graph )
{
GraphNode curr;
GraphNode dummy;
    
  d_visit.init ( graph, 0 );
  
  /*  if (comp_number) {
    comp_number->init (G);
  }
  
  if (preds) {
    preds->init (G, node());
  }
  
  if (back_edges) {
    used = new edge_map<int> (G, 0);
  }
  
  init_handler (G);
  
  //
  // Set start-node 
  // 
  
  if (G.number_of_nodes() == 0) {
    return GTL_OK;
  }
  
  if (start == node()) {
    start = G.choose_node();
  } 
  
  new_start_handler (G, start);
  */
  dfs( graph, d_start, dummy );
  
  /*  if (whole_graph && reached_nodes < G.number_of_nodes()) {
    
    //
    // Continue DFS with next unused node.
    //
    
    forall_nodes (curr, G) {
      if (dfs_number[curr] == 0) {
	new_start_handler (G, curr);
	dfs_sub (G, curr, dummy);
      }
    }
  }    
  
  if (back_edges) {
    delete used;
    used = 0;
  }
  
  end_handler(G);
  
  return GTL_OK;
  */
}

void Dfs::dfs (Graph& graph, GraphNode& curr, GraphNode& father) 
{
GraphNode opp;
GraphEdge adj;
  
/*  if (father == node()) {	
    roots.push_back (dfs_order.insert (dfs_order.end(), curr));
  } else {
    dfs_order.push_back (curr);    
  }
*/  
  d_visit[curr] = d_dfsnum++;
  /*  reached_nodes++;
  
  if (preds) {
    (*preds)[curr] = father;
  }
  
  entry_handler (G, curr, father);
  
  ++act_dfs_num;*/
  GraphNode::adj_edges_iterator it = curr.adj_edges_begin();
  GraphNode::adj_edges_iterator end = curr.adj_edges_end();
  
  while (it != end) {
    adj = *it;
    opp = curr.opposite(adj);
    
    if (d_visit[opp] == 0) { // not visited	    
      /*      tree.push_back (adj);
      
      if (back_edges) {
	(*used)[adj] = 1;
	}*/
      pre_recusive_handler( graph, adj, opp );
      dfs( graph, opp, curr);
      //      post_recursive_handler( graph, adj, opp );
      
    }/* else {
      if (back_edges && !(*used)[adj]) {
	(*used)[adj] = 1;
	back_edges->push_back (adj);
      }
      
      old_adj_node_handler (G, adj, opp);
      }*/
    
    ++it;
  }
  
  /* leave_handler (G, curr, father);
  
  if (comp_number) {
    (*comp_number)[curr] = act_comp_num;
    ++act_comp_num;
    }*/
}

}

#endif






