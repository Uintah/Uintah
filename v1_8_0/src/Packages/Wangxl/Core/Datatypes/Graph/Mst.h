#ifndef SCI_Wangxl_Datatypes_Mst_h
#define SCI_Wangxl_Datatypes_Mst_h

#include "Graph.h"
#include "EdgeMap.h"

#include <vector>
#include <queue>
#include <utility>

namespace Wangxl {

using std::vector;
using std::priority_queue;
using std::pair;

class Mst {
public:
  Mst(){}
  ~Mst(){}
  void run( Graph& graph, EdgeMap< double >& weights);
  class Disjointset {
  public:
    Disjointset(){}
    void print();
    void init( const unsigned int num );
    bool find( unsigned int x, unsigned int y ); 
    void release();
  private:
    int* dad;
  };
  class compare { // comparing function used for edges priority_queue
  public:
    bool operator() ( pair< GraphEdge, double >& ew0,pair< GraphEdge, double > & ew1 )
      {
	return ew0.second > ew1.second;
      }
  };
};

void Mst::run( Graph& graph, EdgeMap< double >& weights )
{
Disjointset nodeSet;
GraphEdge edge;
priority_queue< pair< GraphEdge, double >, vector< pair< GraphEdge, double > >, compare > edgeQueue;
Graph::edge_iterator i;
int source, target;
pair< GraphEdge, double > ew;

  nodeSet.init( graph.getNodesNum() );
  for ( i = graph.edges_begin(); i != graph.edges_end(); i++ ) {
    edge = *i;
    ew.first = edge;;
    ew.second = weights[edge];
    edgeQueue.push( ew );
    //    cout << edge << " " << ew.second << endl;
  }

  while ( !edgeQueue.empty() ) {
    ew = edgeQueue.top();
    edge = ew.first;
    source = edge.source().id();
    target = edge.target().id();
    // ++ because index could not be 0 in Union/Find algorithm
    if ( !nodeSet.find( ++source, ++target ) ) graph.hideEdge( edge );
    edgeQueue.pop();
  }
  nodeSet.release();
}

void Mst::Disjointset::init( const unsigned int num )
{
unsigned int i;

  dad = new int[ num+1 ];
  for ( i = 0; i <= num; i++ ) dad[ i ] = 0; 
}
void Mst::Disjointset::release() { delete [] dad; }
/*void Disjointset::print () {
  for ( int i=0; i <= 25; i++ ) {
    cout << dad[i] << " ";
    cout << endl; 
  }
  }*/
bool Mst::Disjointset::find( unsigned int x, unsigned int y )
{
int tmp, i = x, j = y;

  while ( dad[ i ] > 0 ) i = dad[ i ];
  while ( dad[ j ] > 0 ) j = dad[ j ];
  while ( dad[ x ] > 0 ) {
    tmp = x;
    x = dad[ x ];
    dad[ tmp ] = i;
  } 
  while ( dad[ y ] > 0 ) {
    tmp = y;
    y = dad[ y ];
    dad[ tmp ] = j;
  } 
  if ( i != j ) {
    if ( dad[ j ] < dad[ i ]) {
      dad[ j ] += dad[ i ] - 1;
      dad[ i ] = j;
    }
    else {
      dad[ i ] += dad[ j ] - 1;
      dad[ j ] = i;
    }
  }
  return (i != j );
}

}

#endif


