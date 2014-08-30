/**
 *
 *
 */

#include <iostream>
#include <string>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graphviz.hpp>

using namespace std;
using namespace boost;


//#define VEC

//====================================================================

template<typename Graph>
void print( std::ostream& os, Graph& g )
{
  typedef typename boost::graph_traits<Graph>::vertex_descriptor Vert;
  typedef typename boost::graph_traits<Graph>::vertex_iterator   VertIter;
  typedef typename boost::graph_traits<Graph>::edge_iterator     EdgeIter;
  typedef typename boost::graph_traits<Graph>::edge_descriptor   Edge;

  os << "Vertices:  ";
  std::pair<VertIter,VertIter> verts = vertices(g);
  for( VertIter iv = verts.first; iv!=verts.second; ++iv ){
    os << g[*iv].name << " ";
  }
  os << endl;

  os << "Edges:";
  std::pair<EdgeIter,EdgeIter> es = edges(g);
  for( EdgeIter ie = es.first; ie!=es.second; ++ie ){
    Vert v1 = boost::source( *ie, g );
    Vert v2 = boost::target( *ie, g );
    os << " (" << g[v1].name << "->" << g[v2].name << ")";
  }
  os << endl;
}

//====================================================================

struct VertInfo {
  VertInfo( const string _name, const int _index )
    : name(_name), index(_index)
  {}
  VertInfo() : name(""), index(-1) {}
  string name;
  int index;
};

//====================================================================

template<typename GraphT>
struct VertWriter{
  typedef typename graph_traits<GraphT>::vertex_descriptor Vertex;
  VertWriter( const GraphT& _g ) : g(_g) {}
  void operator()( ostream& os, const Vertex& v ) const{
    os << "[label=\"" << g[v].name << "\"]";
  }
private:
  const GraphT& g;
};

//====================================================================

template<typename G1, typename G2>
void convert_vec2list( const G1& g1, G2& g2 )
{
  typedef typename boost::graph_traits<G1>::vertex_descriptor SrcVert;
  typedef typename boost::graph_traits<G1>::vertex_iterator   SrcVertIter;
  typedef typename boost::graph_traits<G2>::vertex_descriptor DestVert;
  typedef typename boost::graph_traits<G2>::vertex_iterator   DestVertIter;

  typedef typename graph_traits<G1>::edge_iterator SrcEdge;

  g2.clear();

  cout << "converting from vector storage to list storage" << endl;

  int vertid=0;
  typedef std::map<int,int> IDVertMap;  IDVertMap id2vert;
  pair<SrcEdge,SrcEdge> edges1 = edges(g1);
  for( SrcEdge ie = edges1.first; ie != edges1.second; ++ie ){
    SrcVert v1 = source( *ie, g1 );
    SrcVert v2 = target( *ie, g1 );
    DestVert vert1, vert2;
    pair<IDVertMap::const_iterator,bool> result = id2vert.insert( make_pair( g1[v1].index, vertid) );
    if( result.second ){
      VertInfo vi( g1[v1] ); vi.index = vertid++;
      vert1 = add_vertex( vi, g2 );
    }
    else
      vert1 = vertex( result.first->second, g2 );
    result = id2vert.insert( make_pair(g1[v2].index, vertid) );
    if( result.second ){
      VertInfo vi( g1[v2] ); vi.index = vertid++;
      vert2 = add_vertex( vi, g2 );
    }
    else
      vert2 = vertex( result.first->second, g2 );
    add_edge( vert1, vert2, g2 );
  }
}

//====================================================================

template<typename G1, typename G2>
void convert_list2vec( const G1& g1, G2& g2 )
{
  typedef typename boost::graph_traits<G1>::vertex_descriptor SrcVert;
  typedef typename boost::graph_traits<G1>::vertex_iterator   SrcVertIter;
  typedef typename boost::graph_traits<G2>::vertex_descriptor DestVert;
  typedef typename boost::graph_traits<G2>::vertex_iterator   DestVertIter;

  typedef typename graph_traits<G1>::edge_iterator SrcEdge;

  g2.clear();

  cout << "converting from list to vector storage" << endl;

  typedef std::map<int,int> IDVertMap;  IDVertMap id2vert;
  int vertix = 0;
  pair<SrcEdge,SrcEdge> edges1 = edges(g1);
  for( SrcEdge ie = edges1.first; ie != edges1.second; ++ie ){
    DestVert vd1, vd2;
    SrcVert vs1 = source( *ie, g1 );
    SrcVert vs2 = target( *ie, g1 );
    // have we seen either vertex yet?
    const VertInfo vi1 = g1[vs1];
    pair<IDVertMap::const_iterator,bool> result = id2vert.insert( make_pair(vi1.index,vertix) );
    if( result.second ){
      VertInfo vi(vi1.name,vertix++);
      vd1 = add_vertex( vi, g2 );
    }
    else
      vd1 = vertex( result.first->second, g2 );
    const VertInfo vi2 = g1[vs2];
    result = id2vert.insert( make_pair(vi2.index,vertix) );
    if( result.second ){
      vd2 = add_vertex( VertInfo(vi2.name,vertix++), g2 );
    }
    else
      vd2 = vertex( result.first->second, g2 );

    add_edge( vd1, vd2, g2 );
  }
}

//====================================================================

template<class Graph>
void prune_vertex( const int id, Graph& g )
{
  typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
  typedef typename boost::graph_traits<Graph>::edge_iterator     EdgeIter;

  Vertex vert = vertex(id,g);
  cout << "removing vertex " << g[vert].name << endl;

  const pair<EdgeIter,EdgeIter> edges1 = edges(g);
  EdgeIter ie = edges1.first;
  for( EdgeIter next = ie;  ie != edges1.second; ie=next ){
    ++next;
    const Vertex v1 = source(*ie,g);
    const Vertex v2 = target(*ie,g);
    if( v1==vert || v2==vert ) remove_edge( *ie, g );
  }

  boost::clear_vertex   ( vert, g );
  boost::remove_vertex  ( vert, g );

  std::cout << "Pruned graph:" << std::endl;
  print( std::cout, g );
}

//====================================================================

int main()
{
  typedef adjacency_list< vecS,  vecS,  directedS, VertInfo > GraphV;
  typedef adjacency_list< listS, listS, directedS, VertInfo > GraphL;

#ifdef VEC
  typedef GraphV Graph;
#else
  typedef GraphL Graph;
#endif

  typedef graph_traits<Graph>::vertex_descriptor Vertex;
  typedef graph_traits<Graph>::vertex_iterator   VertexIter;

  typedef pair<int,int> Pair;
  Pair edge_array[10] = { Pair(0,1), Pair(0,2), Pair(0,3), 
			  Pair(0,4), Pair(2,3),
			  Pair(2,4), Pair(1,3), Pair(3,4), 
			  Pair(1,2), Pair(1,4) };
  
  vector<string> names;
  names.push_back( "A" );
  names.push_back( "B" );
  names.push_back( "C" );
  names.push_back( "D" );
  names.push_back( "E" );


#ifdef VEC  // vecS map for vertices.

  GraphV GV;

  typedef std::set<int> IDSet;  IDSet ids;
  int vertix = 0;
  for (int i = 0; i < 10; ++i){
    Vertex v1, v2;
    const int i1 = edge_array[i].first;
    const int i2 = edge_array[i].second;
    int vert1, vert2;

    std::pair<IDSet::iterator,bool> result = ids.insert( i1 );
    if( result.second ){
      v1 = add_vertex( VertInfo(names[i1],vertix++), GV );
    }
    else{
      v1 = vertex( *result.first, GV );
    }

    result = ids.insert( i2 );
    if( result.second ){
      v2 = add_vertex( VertInfo(names[i2],vertix++), GV );
    }
    else{
      v2 = vertex( *result.first, GV );
    }
    add_edge( v1, v2, GV);
  } 

  GraphL GL;
  convert_vec2list( GV, GL );
  prune_vertex( 3, GL );
  convert_list2vec( GL, GV );

#else  // listS or setS map for vertices

  GraphL GL;

  typedef std::map<int,int> IDVertMap;  IDVertMap id2vert;
  int vertid=0;
  for (int i = 0; i < 10; ++i){
    const int i1 = edge_array[i].first;
    const int i2 = edge_array[i].second;
    Vertex v1, v2;
    pair<IDVertMap::iterator,bool> result = id2vert.insert( make_pair(i1,vertid) );
    if( result.second ){
      v1 = add_vertex( VertInfo(names[i1],vertid++), GL );
    }
    else
      v1 = vertex( id2vert[i1], GL );
    result = id2vert.insert( make_pair(i2,vertid) );
    if( result.second ){
      v2 = add_vertex( VertInfo(names[i2],vertid++), GL );
    }
    else
      v2 = vertex( id2vert[i2], GL );
    add_edge( v1, v2, GL );
  }

  // vertex removal (with associated edges)
  prune_vertex(2,GL);

  GraphV GV;
  convert_list2vec( GL, GV );

#endif


  std::ofstream fout( "tmp.dot" );
  write_graphviz( fout, GV, VertWriter<GraphV>(GV) );
  fout.close();

  print( cout, GV );
  print( cout, GL );


  vector<string> sortNames1, sortNames2;

//// this was making the new Mac OS (Mavericks) unhappy for some reason.
//  typedef boost::graph_traits<GraphL>::vertex_descriptor GLVert;
//  typedef std::vector< GLVert > container1;  container1 c1;
//  topological_sort( GL,
//		    back_inserter(c1),
//		    vertex_index_map(get(&VertInfo::index,GL)) );
//
//  cout << "Bottom-up ordering: ";
//  for( container1::const_iterator i=c1.begin(); i!=c1.end(); ++i ){
//    cout << GL[*i].name << "  ";
//    sortNames1.push_back( GL[*i].name );
//  }
//  cout << endl;

  typedef boost::graph_traits<GraphV>::vertex_descriptor GVVert;
  typedef std::vector< GVVert > container2;  container2 c2;
  topological_sort( GV,
		    back_inserter(c2),
		    vertex_index_map(get(&VertInfo::index,GV)) );

  for( container2::const_iterator i=c2.begin(); i!=c2.end(); ++i ){
    sortNames2.push_back( GV[*i].name );
  }

//  bool failed = false;
//  for( vector<string>::const_iterator i1=sortNames1.begin(), i2=sortNames2.begin();
//       i1!=sortNames1.end(); ++i1, ++i2 ){
//    failed = (*i1 != *i2);
//  }

//  if( failed ) cout << endl << "FAIL!" << endl;
//  assert( !failed );


}
