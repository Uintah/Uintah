/*
 * Copyright (c) 2011 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//Debug Flags : DEBUG_LOCK_ALL_FIELDS ( default undefined )
//#define DEBUG_LOCK_ALL_FIELDS

/* --- standard includes --- */
#include <stdexcept>
#include <sstream>
#include <ostream>
#include <algorithm>
#include <iomanip>

/* --- Boost includes --- */
#include <boost/graph/graphviz.hpp>
#include <boost/graph/transpose_graph.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

/* --- Expression includes --- */
#include <expression/ExpressionTree.h>
#include <expression/ExpressionFactory.h>
#include <expression/ExpressionBase.h>
#include <expression/FieldManagerList.h>
#include <expression/ManagerTypes.h>
#include <expression/ExprDeps.h>
#include <expression/Tag.h>

#ifdef ENABLE_THREADS
#include <spatialops/ThreadPool.h>
#endif

#include <spatialops/structured/MemoryTypes.h>

using std::cout;
using std::endl;

//#define PRUNE_DIAGNOSTICS

namespace Expr{

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
    const VertexProperty& vp = g[*iv];
    const TagList& names = vp.expr->get_tags();
    BOOST_FOREACH( const Tag& t, names ){
      os << t.field_name() << " ";
    }
    os << " (ix=" << vp.index << ") ";
  }
  os << endl;

  os << "Edges:";
  std::pair<EdgeIter,EdgeIter> es = edges(g);
  for( EdgeIter ie = es.first; ie!=es.second; ++ie ){
    Vert v1 = boost::source( *ie, g );
    Vert v2 = boost::target( *ie, g );
    os << " (" << g[v1].expr->get_tags()[0].field_name() << "->" << g[v2].expr->get_tags()[0].field_name() << ")";
  }
  os << std::endl;
}

//------------------------------------------------------------------

class CleaveVisitor : public boost::default_bfs_visitor
{
  const ExpressionTree::IDSet& tagged_;
  ExpressionTree::IDSet& buildList_;
  ExpressionTree::IDSet& haveSeen_;

public:

  /**
   * @brief Used to determine vertices in child graphs resulting from cleaving
   * @param taggedIDs The set of IDs that we want to cleave.
   * @param haveSeen A set of tagged IDs that have been seen already during the
   *        search, indicating that there are multiple pathways to get to these
   *        through the graph we are searching.  Note that this set can have
   *        values pre-populated, which results in discovered nodes not being
   *        added to the buildList. Only nodes that have not been seen and are
   *        tagged will be added to the buildList.
   * @param buildList The list of tagged vertices that are found in the graph
   *        and are not already in the haveSeen set.
   */
  CleaveVisitor( const ExpressionTree::IDSet& taggedIDs,
                 ExpressionTree::IDSet& haveSeen,
                 ExpressionTree::IDSet& buildList )
    : tagged_( taggedIDs ),
      buildList_( buildList ),
      haveSeen_( haveSeen )
  {}

  void examine_vertex( const Vertex& v,
                       const Graph& g )
  {
    const VertexProperty& vp = g[v];
    const ExpressionID& id = vp.id;

    // see if this is a node we have tagged.  If it is, then we
    // add it to the build list unless we have already visited it.
    if( tagged_.find(id) != tagged_.end() ){
      ExpressionTree::IDSet::iterator ibl = buildList_.find(id);
      if( haveSeen_.find(id) != haveSeen_.end() ){
        // already seen - remove from build list, since this will be a
        // root node of a descendant tree.
        buildList_.erase( id );
      }
      else{
        buildList_.insert( id );
      }
      haveSeen_.insert( id );
    }
  }

};

//------------------------------------------------------------------

class FieldPruneVisitor : public boost::default_bfs_visitor
{
  ExpressionTree::IDSet& retainedFields_;
  const ExpressionTree::IDSet& removalFields_;
public:
  FieldPruneVisitor( ExpressionTree::IDSet& retained,
                     const ExpressionTree::IDSet& toBePruned )
    : retainedFields_( retained ),
      removalFields_( toBePruned )
  {}

  void examine_edge( const Edge& e,
                     const Graph& g )
  {
    // if a field on the "source" side of the edge has not been tagged
    // for removal, then retain the source and destination fields.
    const ExpressionID sid = g[source(e,g)].id;
    if( removalFields_.find( sid ) == removalFields_.end() ){
      const ExpressionID tid = g[target(e,g)].id;
      retainedFields_.insert( sid );
      retainedFields_.insert( tid );
    }
  }
};

//--------------------------------------------------------------------

void cleave_tree( ExpressionTree::TreePtr parent,
                  ExpressionTree::IDSet& taggedList,
                  ExpressionTree::TreeList& treeList )
{
  if( parent->is_cleaved() ) return;
  parent->isCleaved_ = true;

  if( taggedList.empty() ){
    // terminate recursion - we are done.
    treeList.insert( treeList.begin(), parent );
    return;
  }

  ExpressionTree::IDSet buildList, haveSeen;
  Graph& graph = *parent->graph_;

  BOOST_FOREACH( const ExpressionID id, taggedList ){
    // resolve this node in the graph and conduct a BFS off of it.
    CleaveVisitor cv( taggedList, haveSeen, buildList );
    boost::breadth_first_search( graph,
                                 parent->find_vertex(graph,id),
                                 boost::color_map( boost::get(&VertexProperty::color,graph)).visitor(cv) );
  }

  // build a child graph containing the points marked for cleaving
  std::string childName;
  {
    const std::string::size_type n = parent->name().find("_child_");
    unsigned childNum = 1;
    if( n != std::string::npos ){
      childNum = boost::lexical_cast<unsigned>( parent->name().substr(n+7) ) + 1;
    }
    childName = parent->name().substr(0,n) + "_child_" + boost::lexical_cast<std::string>(childNum);
  }
  ExpressionTree::TreePtr child( new ExpressionTree( buildList,
                                                     parent->get_expression_factory(),
                                                     parent->patch_id(),
                                                     childName ) );

  // remove the cleaved nodes from the list of tagged nodes remaining to be cleaved.
  BOOST_FOREACH( const ExpressionID id, buildList ){
    taggedList.erase( id );
  }

  // prune the child graph from the parent and then push the parent onto the treeList
  parent->prune( child );
  treeList.insert( treeList.begin(), parent );

  // now recurse onto the child graph to cleave it further if required
  cleave_tree( child, taggedList, treeList );
}

//===================================================================

ExpressionTree::ExpressionTree( const ExpressionID rootID,
                                ExpressionFactory& exprFactory,
                                const int patchid,
                                const std::string name )
  : name_( name ),
    patchID_( patchid ),
    factory_( exprFactory ),
    bUpdatePScore_(false),
    bTimingsReady_(false),
    graph_ ( NULL ),
    graphT_( NULL )
{
# ifdef ENABLE_CUDA
  cudaError err;
  deviceID_        = 0;
  deviceAlreadySet_  = false;
# endif
  hasRegisteredFields_ = false;
  hasBoundFields_      = false;
  isCleaved_           = false;
  rootIDs_.insert( rootID );

  compile_expression_tree();
}

//--------------------------------------------------------------------

ExpressionTree::ExpressionTree( const RootIDList& ids,
                                ExpressionFactory& factory,
                                const int patchid,
                                const std::string name )
  : name_( name ),
    patchID_( patchid ),
    rootIDs_( ids.begin(), ids.end() ),
    factory_( factory ),
    graph_ ( NULL ),
    graphT_( NULL )
{
# ifdef ENABLE_CUDA
  cudaError err;
  deviceID_        = 0;
  deviceAlreadySet_  = false;
# endif
  bUpdatePScore_       = false;
  bTimingsReady_       = false;
  hasRegisteredFields_ = false;
  hasBoundFields_      = false;
  isCleaved_           = false;

  compile_expression_tree();
}

//--------------------------------------------------------------------

ExpressionTree::ExpressionTree( ExpressionFactory& exprFactory,
                                const int patchid,
                                const std::string name )
  : name_( name ),
    patchID_( patchid ),
    factory_( exprFactory ),
    graph_(NULL),
    graphT_(NULL)
{
# ifdef ENABLE_CUDA
  deviceID_        = 0;
  deviceAlreadySet_  = false;
# endif
  bUpdatePScore_       = false;
  bTimingsReady_       = false;
  hasRegisteredFields_ = false;
  hasBoundFields_      = false;
  isCleaved_           = false;
}

//--------------------------------------------------------------------

ExpressionTree::~ExpressionTree()
{
  delete graph_;
  delete graphT_;
}

//--------------------------------------------------------------------

void
ExpressionTree::insert_tree( const IDSet& ids )
{
  BOOST_FOREACH( const ExpressionID id, ids ){
    rootIDs_.insert( id );
  }
  hasRegisteredFields_ = false;
  compile_expression_tree();
}

//--------------------------------------------------------------------

void
ExpressionTree::insert_tree( const ExpressionID rootID )
{
  rootIDs_.insert( rootID );
  hasRegisteredFields_ = false;

  compile_expression_tree();
}

//--------------------------------------------------------------------

void
ExpressionTree::insert_tree( ExpressionTree& tree )
{
  const RootIDList& ids = tree.get_roots();
  BOOST_FOREACH( const ExpressionID id, ids ){
    rootIDs_.insert( id );
  }
  compile_expression_tree();
}

//--------------------------------------------------------------------

void
ExpressionTree::execute_tree()
{
  if( !hasBoundFields_ ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << std::endl
        << "ERROR! Tree named '" << name() << "' has not yet bound fields" << std::endl;
  }
  if( !hasRegisteredFields_ ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << std::endl
        << "ERROR! Tree named '" << name() << "' has not yet registered fields" << std::endl;
  }

  // be sure that we have a graph built.
  if( graph_ == NULL ){
    std::ostringstream msg;
    msg << __FILE__ << ":" << __LINE__ << std::endl
        << "ERROR!  Cannot execute the tree with root set: " << std::endl;
    BOOST_FOREACH( const ExpressionID id, rootIDs_ ){
      msg << "        "  << id << ", " << factory_.get_labels(id) << std::endl;
    }
    msg << "  because the graph does not exist." << std::endl
	<< "  You must first run compile_expression_tree()" << std::endl;
    throw std::runtime_error( msg.str() );
  }

  scheduler_->setup( hasRegisteredFields_ );

  if( !bUpdatePScore_ ){
    scheduler_->run();
  }
  else{
    scheduler_->run();
    bTimingsReady_ = true;
    push_schedule_timings();
    bUpdatePScore_ = false;
  }
}

//--------------------------------------------------------------------

void
ExpressionTree::register_fields( FieldManagerList& fml )
{
  // jcs deprecate use?
  FMLMap fmls; fmls[0]=&fml;
  register_fields(fmls);
}

void
ExpressionTree::register_fields( FMLMap& fmls )
{
  for( ExprFieldMap::iterator iefm = exprFieldMap_.begin(); iefm!=exprFieldMap_.end(); ++iefm ){

    FieldDeps& fd = *(iefm->second);

    // on cleaving, we may not have a vertex for the "shadow" fields.
    const ID2Vert::const_iterator ievmT = exprVertexMapT_.find( iefm->first );
    if( ievmT != exprVertexMapT_.end() ) {

      VertexProperty& vpT = (*graphT_)[ievmT->second];
      VertexProperty& vp  = (*graph_)[ find_vertex(*graph_,vpT.id) ];

      // CARRY_FORWARD and DYNAMIC fields must be locked to allow their values to be "remembered"
      // jcs what about the possibility of other tags being on the list???
      const Expr::TagList& tags = factory_.get_labels(iefm->first);
      if( tags[0].context() == CARRY_FORWARD || tags[0].context() == STATE_DYNAMIC ){
        vpT.set_is_persistent(true);
        vp .set_is_persistent(true);
      }

      FieldManagerList& fml = *extract_field_manager_list( fmls, vpT.fmlid );
      fd.register_fields( fml );
      fd.set_memory_manager( fml, vpT.mm, vpT.deviceIndex_ );
    }
    else if( fmls.size() == 1 ){
      // when cleaving, we need to register the first level cleaved fields
      // (out-edges) however, this will not function properly with multiple
      // FieldManagerList objects since we do not have a way to determine
      // which FML the fields should come off of.
      fd.register_fields( *fmls[0] );
    }
    else{
      // The only way we get here is if we have a field that we depend on but no
      // corresponding expression in the graph.  This should only be due to cleaving.
      // In that case, a cleaved edge retains the field dependency but no the
      // actual expression, which is contained in a child graph.  To fix this,
      // we would need to include FieldManagerList IDs in with the FieldDeps
      // so we knew which FML to pull the field off of.
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "ERROR from: ExpressionTree::register_fields()" << std::endl
          << "  When using multiple FieldManagerList objects, cleaving is not supported" << std::endl;
      throw std::runtime_error(msg.str());
    }
  }

# ifdef DEBUG_LOCK_ALL_FIELDS
  std::cout << "Locking all fields" << std::endl;
  lock_fields(fmls);
# endif

  hasRegisteredFields_ = true;
  scheduler_->set_fmls( fmls );
  scheduler_->set_fdm( &exprFieldMap_ );

  scheduler_->setup( hasRegisteredFields_ );
}

//--------------------------------------------------------------------

void
ExpressionTree::lock_fields( FMLMap& fmls ){
  BOOST_FOREACH( FMLMap::value_type& i, fmls ){ lock_fields(*i.second); }
}

void
ExpressionTree::lock_fields( FieldManagerList& fml )
{
  for( ExprFieldMap::iterator iefm = exprFieldMap_.begin(); iefm!=exprFieldMap_.end(); ++iefm ){
    FieldDeps& fd = *(iefm->second);
    fd.lock_fields( fml );
  }
}

//--------------------------------------------------------------------

void
ExpressionTree::unlock_fields( FMLMap& fmls ){
  BOOST_FOREACH( FMLMap::value_type& i, fmls ){ unlock_fields(*i.second); }
}

void
ExpressionTree::unlock_fields( FieldManagerList& fml )
{
  for( ExprFieldMap::iterator iefm = exprFieldMap_.begin(); iefm!=exprFieldMap_.end(); ++iefm ){
    FieldDeps& fd = *(iefm->second);
    fd.unlock_fields( fml );
  }
}

//--------------------------------------------------------------------
void
ExpressionTree::bind_fields( FieldManagerList& fml )
{
  // be sure that we have a graph built.
  if( graph_ == NULL ){
    std::ostringstream msg;
    msg << __FILE__ << ":" << __LINE__ << std::endl
        << "ERROR!  Cannot bind fields on the tree with root set: " << std::endl;
    BOOST_FOREACH( const ExpressionID id, rootIDs_ ){
      msg << "        "  << id << ", " << factory_.get_labels(id) << std::endl;
    }
    msg << "  because the graph does not exist." << std::endl
	<< "  You must first run compile_expression_tree()" << std::endl;
    throw std::runtime_error( msg.str() );
  }

  FMLMap fmls; fmls[0]=&fml;
  scheduler_->set_fmls(fmls);
  scheduler_->set_fdm(&exprFieldMap_);
  hasBoundFields_ = true;
}
void
ExpressionTree::bind_fields( FMLMap& fmls )
{
  // be sure that we have a graph built.
  if( graph_ == NULL ){
    std::ostringstream msg;
    msg << __FILE__ << ":" << __LINE__ << std::endl
        << "ERROR!  Cannot bind fields on the tree with root set: " << std::endl;
    BOOST_FOREACH( const ExpressionID id, rootIDs_ ){
      msg << "        "  << id << ", " << factory_.get_labels(id) << std::endl;
    }
    msg << "  because the graph does not exist." << std::endl
	<< "  You must first run compile_expression_tree()" << std::endl;
    throw std::runtime_error( msg.str() );
  }

  scheduler_->set_fmls(fmls);
  scheduler_->set_fdm(&exprFieldMap_);
  hasBoundFields_ = true;
}

//--------------------------------------------------------------------

void
ExpressionTree::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  OpDBMap opDBs; opDBs[0] = &opDB;
  bind_operators( opDBs );
}

//--------------------------------------------------------------------

void
ExpressionTree::bind_operators( const OpDBMap& opDBs )
{
  // be sure that we have a graph built.
  if( graph_ == NULL ){
    std::ostringstream msg;
    msg << __FILE__ << ":" << __LINE__ << std::endl
        << "ERROR!  Cannot bind operators on tree with root set: " << std::endl;
    BOOST_FOREACH( const ExpressionID id, rootIDs_ ){
      msg << "        "  << id << ", " << factory_.get_labels(id) << std::endl;
    }
    msg << "  because the graph does not exist." << std::endl
	<< "  You must first run compile_expression_tree()" << std::endl;
    throw std::runtime_error( msg.str() );
  }

  const std::pair<VertIter,VertIter> verts = vertices(*graph_);
  for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
    ExpressionBase* const expr = (*graph_)[*iv].expr;
    const SpatialOps::OperatorDatabase* opDB = opDBs.begin()->second;
    if( opDBs.size() > 1 ){
      const int fmlID = expr->field_manager_list_id();
      const OpDBMap::const_iterator iop = opDBs.find(fmlID);
      assert( iop != opDBs.end() );
      opDB = iop->second;
    }
    expr->base_bind_operators( *opDB );
  }
}

//--------------------------------------------------------------------

void
ExpressionTree::compile_expression_tree()
{
  tOne_ = 0; tINF_ = 0;
  exprVertexMapT_.clear();
  delete graph_;
  delete graphT_;

  graph_  = new Graph();
  graphT_ = new Graph();

  try{
    ID2Index id2index;
    int vertIx=0;
    int depth = 0;
    BOOST_FOREACH( const ExpressionID iid, rootIDs_ ){
      bootstrap_dependents( iid, vertIx, id2index, depth );
    }
  }
  catch( std::runtime_error& err ){
    std::ostringstream msg;
    msg << __FILE__ << ":" << __LINE__ << std::endl
        << "Fatal error constructing expression tree."
        << std::endl
        << err.what() << std::endl;
    throw std::runtime_error( msg.str() );
  }

  boost::transpose_graph( *graph_, *graphT_,
                          boost::vertex_index_map( boost::get(&VertexProperty::index, *graph_) ) );

  BOOST_FOREACH( PollerPtr p, factory_.get_pollers() ){
    if( has_expression( p->target_tag() ) ){
      pollers_.insert( p );
      // push onto vertex properties
      Graph& gT = *graphT_;
      const Vertex vT = find_vertex( gT, get_id(p->target_tag()), true );
      VertexProperty& vpT = gT[ vT ];
      p->deactivate_all();  // jcs should be deactivated by default and activated when a node completes.
      vpT.poller = p;
      assert( vpT.poller );

      Graph& g = *graph_;
      const Vertex v = find_vertex( g, get_id(p->target_tag()), true );
      VertexProperty& vp = g[v];
      vp.poller = p;
      assert( vp.poller );
    }
  }
  BOOST_FOREACH( NonBlockingPollerPtr p, factory_.get_nonblocking_pollers() ){
    if( has_expression( p->target_tag() ) ){
      nonBlockPollers_.insert( p );
      // push onto vertex properties
      Graph& gT = *graphT_;
      const Vertex vT = find_vertex( gT, get_id(p->target_tag()), true );
      VertexProperty& vpT = gT[ vT ];
      p->deactivate_all();  // jcs should be deactivated by default and activated when a node completes.
      vpT.nonBlockPoller = p;
      assert( vpT.nonBlockPoller );

      Graph& g = *graph_;
      const Vertex v = find_vertex( g, get_id(p->target_tag()), true );
      VertexProperty& vp = g[v];
      vp.nonBlockPoller = p;
      assert( vp.nonBlockPoller );
    }
  }

# ifdef ENABLE_CUDA
  scheduler_.reset( new HybridScheduler(*graph_,*graphT_) );
# else
  scheduler_.reset( new PriorityScheduler(*graph_,*graphT_) );
# endif

  BOOST_FOREACH( PollerPtr p, factory_.get_pollers() ){
    scheduler_->set_poller(p);
  }
  BOOST_FOREACH( NonBlockingPollerPtr p, factory_.get_nonblocking_pollers() ){
    scheduler_->set_nonblocking_poller(p);
  }

  // Setup expression => vertex property map
  {
    std::pair<VertIter,VertIter> verts = boost::vertices(*graphT_);
    for( VertIter iv = verts.first; iv != verts.second; ++iv ){
      exprVertexMapT_.insert( std::make_pair( (*graphT_)[*iv].id, *iv ) );
    }
  }

  // ensure that all root nodes are actually roots now that we have the graph built.
  {
    for( RootIDList::const_iterator iid=rootIDs_.begin(); iid!=rootIDs_.end(); ++iid ){
      if( boost::out_degree( exprVertexMapT_[*iid], *graphT_ ) > 0 ){
        if( factory_.is_logging_active() ){
          std::cout << "WARNING: Node labeled '" << factory_.get_labels(*iid)[0] << "'" << std::endl
                    << "  was set as a root node but has in-edges." << std::endl
                    << "  Removing from root id list." << std::endl << std::endl;
        }
        rootIDs_.erase(iid);
        iid = rootIDs_.begin();  // jcs on some platforms this was required to avoid segfaults in the increment operator
      }
    }
  }

}

//--------------------------------------------------------------------

void
ExpressionTree::bootstrap_dependents( const ExpressionID& id,
                                      int& vertIx,
                                      ID2Index& id2index,
                                      int& depth )
{
  const int MAX_DEPTH = 30;
  if( depth>MAX_DEPTH ){
    std::ostringstream msg;
    msg << __FILE__ << ":" << __LINE__ << std::endl
        << "ERROR! Maximum recursion depth (" << MAX_DEPTH << ") exceeded in the tree." << std::endl
        << "       This may indicate a circular dependency in the graph" << std::endl
        << "       The current graph is in 'graph_problems.dot'" << std::endl
        << std::endl
        << "       ROOT ID(s) follow:" << std::endl;
    BOOST_FOREACH( const ExpressionID id, rootIDs_ ){
      msg << "       " << id << ",   "
          << factory_.get_labels(id) << std::endl;
    }
    std::ofstream fout( "graph_problems.dot" );
    write_tree( fout );
    throw std::runtime_error( msg.str() );
  }

  if( id == ExpressionID::null_id() ){
    std::ostringstream msg;
    msg << __LINE__ << " : " << __FILE__ << endl
        << "Invalid ExpressionID (null ID) detected in the tree." << endl;
    throw std::runtime_error( msg.str() );
  }

  //
  // 1. obtain the expression corresponding to this ID
  //
  ExpressionBase& expr = factory_.retrieve_expression(id,patchID_,false);

  //
  // 2. Determine its dependencies
  //
  ExprDeps exprDeps;
  boost::shared_ptr<FieldDeps> fieldDeps( new FieldDeps() );
  try{
    expr.base_advertise_dependents( exprDeps, *fieldDeps );
    exprFieldMap_[id] = fieldDeps;
  }
  catch( std::runtime_error& e ){
    std::ostringstream msg;
    msg << "ERROR analyzing dependents of expression " << expr.get_tags()[0] << std::endl
        << e.what() << std::endl;
    throw std::runtime_error( msg.str() );
  }

  //
  // 3. Insert this information into the graph
  //
  // jcs should this be done in the main compile_ function?
  bool parentAlreadyPresent;
  Vertex parentVertex;
  add_vertex_to_graph( parentVertex, parentAlreadyPresent, vertIx, id2index, id );

  if( exprDeps.begin() == exprDeps.end() ){
    (*graph_)[parentVertex].set_is_persistent(true);
  }

  // iterate through the children
  for( ExprDeps::const_iterator iexd=exprDeps.begin(); iexd!=exprDeps.end(); ++iexd ){

    Vertex childVertex;
    try{
      const ExpressionID childID = factory_.get_id(*iexd);
      bool childAlreadyPresent;
      add_vertex_to_graph( childVertex, childAlreadyPresent, vertIx, id2index, childID );
      { // add this edge?
        bool edgeExists = false;
        typedef boost::graph_traits<Graph>::adjacency_iterator AdjIter;
        typedef std::pair<AdjIter,AdjIter> AdjVerts;
        const AdjVerts verts = boost::adjacent_vertices( parentVertex, *graph_ );
        for( AdjIter iv=verts.first; iv!=verts.second; ++iv ){
          if( *iv == childVertex ){
            edgeExists = true;
            break;
          }
        }
        if( !edgeExists ){
          boost::add_edge( parentVertex, childVertex, *graph_ );
          // "interior" nodes are set as not persistent unless over-ridden elsewhere.
          VertexProperty& vp = (*graph_)[childVertex];
          vp.set_is_persistent(false);
        }
      }
      if( !childAlreadyPresent ){
        //
        // 4. Recurse on each of this child
        //
        bootstrap_dependents( childID, vertIx, id2index, ++depth );
      }
    }
    catch( std::runtime_error& e ){
      std::ostringstream msg;
      msg << "ERROR: while examining dependents of '" << factory_.get_labels(id)[0] << "'" << std::endl
          << "       could not resolve expression '" << *iexd << "'" << std::endl
          << std::endl << e.what() << std::endl
          << std::endl << "all advertised dependents of '" << factory_.get_labels(id)[0] << "' follow:" << std::endl;
      for( ExprDeps::const_iterator ii=exprDeps.begin(); ii!=exprDeps.end(); ++ii ){
        msg << "  -> " << *ii << std::endl;
      }
      throw std::runtime_error( msg.str() );
    }
  }
  --depth;
}

//--------------------------------------------------------------------

void
ExpressionTree::add_vertex_to_graph( Vertex& v,
                                     bool& alreadyPresent,
                                     int& vertIx,
                                     ID2Index& id2index,
                                     const ExpressionID& id )
{
  std::pair<ID2Index::const_iterator,bool> result = id2index.insert( std::make_pair(id,vertIx) );
  if( result.second ){
    ExpressionBase& expr = factory_.retrieve_expression(id,patchID_);
    const VertexProperty vpsrc( vertIx++, id, &expr );
    v = boost::add_vertex( vpsrc, *graph_ );
    alreadyPresent = false;
  }
  else{
    alreadyPresent = true;
    v = find_vertex( *graph_, id, true );
  }
}

//--------------------------------------------------------------------

ExpressionTree::TreeList
ExpressionTree::split_tree()
{
  /******* Algorithm *******

    1. Collect a list of tagged expressions for nodes that are to
       become new "root" nodes (out-edges from cleaved edge)

    2. Determine the subset of tagged IDs that will become the root
       nodes of the child graph

        - construct a child graph from the subset of tagged IDs

        - remove the id subset from the tagged list

        - any nodes that are in the child graph and have edges that were cleaved
          are marked as persistent, since they will be required for use later on.

    3. prune the child graph from the parent graph.

    4. recurse to step 2 with the child becoming the parent

   ******* Algorithm *******/

  // look for nodes tagged to be cleaved
  IDSet taggedIDs;
# ifdef PRUNE_DIAGNOSTICS
  cout << endl << "The following nodes have been marked for cleaving: " << endl;
# endif // PRUNE_DIAGNOSTICS
  const std::pair<VertIter,VertIter> verts = vertices(*graph_);
  for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
    VertexProperty& vp = (*graph_)[*iv];
    if( vp.expr->cleave_from_parents() ){
      vp.set_is_persistent(true);
      // Make sure node is not a root
      if( std::find( rootIDs_.begin(), rootIDs_.end(), vp.id ) == rootIDs_.end() ){
        // this id forms a root in the child graph
        const std::pair<IDSet::iterator,bool> it = taggedIDs.insert( vp.id );
#       ifdef PRUNE_DIAGNOSTICS
        if( it.second )
          cout << "   " << factory_.get_labels(vp.id) << endl;
#       endif // PRUNE_DIAGNOSTICS
      }
    }

    if( vp.expr->cleave_from_children() ){
      // children may form roots of the new graph.  In some cases, however,
      // a child may be an interior node in a resulting subgraph.
      typedef boost::graph_traits<Graph>::out_edge_iterator OutEdgeIter;
      std::pair<OutEdgeIter,OutEdgeIter> es = boost::out_edges(*iv,*graph_);
      for( OutEdgeIter ie = es.first; ie!=es.second; ++ie ){
        const Vertex& outv = boost::target( *ie, *graph_ );
        VertexProperty& outvp = (*graph_)[outv];
        outvp.set_is_persistent(true);
        const std::pair<IDSet::iterator,bool> it = taggedIDs.insert( outvp.id );
#       ifdef PRUNE_DIAGNOSTICS
        if( it.second )
          cout << "   " << factory_.get_labels(outvp.id) << endl;
#       endif
      }
    }
  }
# ifdef PRUNE_DIAGNOSTICS
  cout << endl;
# endif // PRUNE_DIAGNOSTICS

  // Build the list of trees
  TreeList treeList;
  TreePtr parent( new ExpressionTree( rootIDs_, factory_, patchID_, name_ ) );

  // Push down persistence properties
  const std::pair<VertIter,VertIter> vertx = vertices(*(parent->graph_));
    for( VertIter iv = vertx.first; iv != vertx.second; iv++ ){
    VertexProperty& vp = (*(parent->graph_))[*iv];

      if( has_expression( vp.id ) ){
        const TagList& tgl = factory_.get_labels( vp.id );
        for( TagList::const_iterator it=tgl.begin(); it!=tgl.end(); ++it ){
          vp.set_is_persistent( is_persistent( *it ) );
        }
      }
    }
  cleave_tree( parent, taggedIDs, treeList );

  return treeList;
}

//--------------------------------------------------------------------

void
ExpressionTree::write_tree( std::ostream& os,
                            const bool execTree,
                            const bool details ) const
{
  assert( graph_!=NULL );
  Graph& g = *graph_;

  os << "digraph {\n";
  if( details ) os << "compound=true;\n";
  os << "node[style=filled]  /* remove this line to have empty (rather than filled) nodes */\n\n";

  // write out vertex (node) information
  os << "\n/*\n * Write out node labels\n */\n";
  const std::pair<VertIter,VertIter> iters = boost::vertices(g);
  for( VertIter iv=iters.first; iv!=iters.second; ++iv  ){
    const VertexProperty& vp = g[*iv];
    os << vp.id << "[";
    if( vp.expr->is_gpu_runnable() ) os << " /* GPU-enabled */ color=darkkhaki,";
    if( vp.expr->is_placeholder()   ) os << " /* placeholder */ shape=diamond,color=powderblue,";
    os << "label=\"";
    const TagList& names = vp.expr->get_tags();
    BOOST_FOREACH( const Expr::Tag& tag, names ){
      if( tag != names[0] )  os << "\n" << tag;
      else                   os << tag << " ";
    }
    const int npp = vp.expr->num_post_processors();
    if( npp > 0 ){
      os << "\n(" << npp << " post-procs)";
    }
    os <<"\"]" << std::endl;
    if( details && vp.expr->num_post_processors() > 0 ){
      os << "\nsubgraph cluster" << vp.id << "{\n"
         << "\t/* this contains modifiers & functors evaluated in conjunction with the expression */\n";
      const std::vector<std::string> names = vp.expr->post_proc_names();
      assert( names.size() == vp.expr->num_post_processors() );
      int imod=1;
      BOOST_FOREACH( const std::string& name, names ){
        os << "\t" << vp.id << "." << imod << "[label=\"" << name << "\"];\n";
        os << "\t" << vp.id << " -> " << vp.id << "." << imod << "\n";
        ++imod;
      }
      os << "}\n\n";
    }

  }

  // write out edge information
  os << "\n/*\n * Write out edge information\n */\n";
  const std::pair<EdgeIter,EdgeIter> edgeIters = boost::edges(g);
  for( EdgeIter ie=edgeIters.first; ie!=edgeIters.second; ++ie ){
    const VertexProperty& svp = g[source(*ie,g)];
    const VertexProperty& dvp = g[target(*ie,g)];
    if( details && svp.expr->num_post_processors() > 0 ){
      os << svp.id << " -> " << dvp.id << " [ltail=cluster" << svp.id << "];\n";
    }
    else{
      os << svp.id << " -> " << dvp.id << std::endl;
    }
  }

  // look for pollers and add those...
  if( pollers_.size() > 0 ) os << "\n/*\n * Poller information augmenting graph\n */\n";
  BOOST_FOREACH( const PollerPtr poller, pollers_ ){
    const VertexProperty& vp = *poller->get_vertex_property();
    os << vp.id << ".1[shape=box,fillcolor=beige,label=\"POLLER\"]\n";
    os << vp.id << " -> " << vp.id << ".1 [style=dotted]\n\n";
    // jcs need to get an out edge from the poller onto the field it polls.
    // but we don't have a way to accomplish this currently...
  }

  os << "\nsubgraph cluster_01 {"
     << "\n label = \"Legend\";\n"
     << "\n a[ /* GPU-enabled */ color=darkkhaki,label=\"GPU Enabled\", weight=0]"
     << "\n b[ label=\"CPU Enabled\", weight=1]"
     << "\n c[ shape=diamond,color=powderblue,label=\"Placeholder\", weight =2]"
     << "\n edge[style=invis];"
     << "\n  a->b->c"
     << "\n}"
    << std::endl;

  os << "\n}\n" << std::endl;
}

//--------------------------------------------------------------------

void
ExpressionTree::dump_vertex_properties( std::ostream& os, const bool execGraph ) const
{
  Graph& g = (execGraph ? *graphT_ : *graph_);
  const std::pair<VertIter,VertIter> iters = boost::vertices(g);
  os << "\n---------------------------------------\n";
  if( execGraph ) os << "Execution Graph Vertex Properties:\n";
  else            os << "Dependency Graph Vertex Properties:\n";
  for( VertIter iv=iters.first; iv!=iters.second; ++iv  )
    os << g[*iv];
  os << "---------------------------------------\n\n";
}

//--------------------------------------------------------------------

#ifdef ENABLE_CUDA

void
ExpressionTree::wait_on_cuda_stream()
{
// This method has to be thread-safe for concurrent task execution on multi-GPUs (multi-threading on
// host is used to lauching the tasks)

  ExecMutex lock;   // thread-safe

  cudaSetDevice(deviceID_);
  cudaError err;
  Graph& g = *graph_;
  bool streamComplete = false;
  const std::pair<VertIter,VertIter> iters = boost::vertices(g);

  for( VertIter iv=iters.first; iv!=iters.second; ++iv ){
    // check is performed for both homo and hetero task graphs
    while( !streamComplete ){
      err = cudaStreamQuery( g[*iv].expr->get_cuda_stream() );
      if     ( err == cudaSuccess      )  streamComplete = true;
      else if( err = cudaErrorNotReady )  streamComplete = false;
      else if( err = cudaErrorInvalidResourceHandle ){
        std::ostringstream msg;
        msg << "ERROR ! Invalid resource handle, might have been created in a different context \n"
            << " at " << __FILE__ << " : " << __LINE__
            << std::endl;
        msg << "\t - " << cudaGetErrorString(err);
        throw(std::runtime_error(msg.str()));
      }
      else{
        std::ostringstream msg;
        msg << "ERROR ! CUDA Asynchronous failure detected from previous launches,  \n"
            << " at " << __FILE__ << " : " << __LINE__
            << std::endl;
        msg << "\t - " << cudaPeekAtLastError();
        throw(std::runtime_error(msg.str()));
      }
    }// while
  }// for
}

//--------------------------------------------------------------------

void
ExpressionTree::set_device_index( const unsigned int deviceIndex, FieldManagerList& fml )
{
  // If a device ID is set from outside of ExprLib like Wasatch, the information should
  // be passed downstream

  SpatialOps::DeviceTypeTools::check_valid_index( deviceIndex, __FILE__, __LINE__ );

  if( !deviceAlreadySet_ ){
    //set the index to the tree
    deviceID_ = deviceIndex;

    // set the device index to the Scheduler
    scheduler_->set_device_index( deviceIndex );
    deviceAlreadySet_ = true;
  }
  else if( deviceID_ != deviceIndex && deviceAlreadySet_ ){
    std::cout << "Warning : device context is changed to ID(" << deviceIndex << ") from the previous ID(" << deviceID_ << ") \n";

    // Release the resources from the old device index (deviceID_)
    fml.deallocate_fields();

    // Allocate or reset the resources for the new device index (deviceIndex)
    deviceID_ = deviceIndex;
    scheduler_->set_device_index( deviceID_ );
    scheduler_->invalidate();
    register_fields( fml );
  }
  else{
    return;
  }
}

//--------------------------------------------------------------------

int
ExpressionTree::get_device_index() const
{
  return deviceID_;
}

#endif // ENABLE_CUDA

//--------------------------------------------------------------------

bool
ExpressionTree::operator==( const ExpressionTree& other ) const
{
  return this->rootIDs_ == other.rootIDs_;
}

//--------------------------------------------------------------------

ExpressionID
ExpressionTree::get_id( const Tag& label ) const
{
  return factory_.get_id(label);
}

//--------------------------------------------------------------------

bool
ExpressionTree::has_expression( const ExpressionID& id ) const
{
  return ( find_vertex(*graph_,id,false) != Vertex() );
}

//--------------------------------------------------------------------

bool
ExpressionTree::has_field( const Tag& tag ) const
{
  if( !factory_.have_entry(tag) ) return false;
  const ExpressionID id = factory_.get_id( tag );
  return ( exprFieldMap_.find(id) != exprFieldMap_.end() );
}

//--------------------------------------------------------------------

bool ExpressionTree::is_persistent( const Tag& tag ) const
{
  if( !has_expression(get_id(tag)) ) return false;

  const ExpressionID id = factory_.get_id(tag);
  const Vertex& v = exprVertexMapT_.find(id)->second;
  const VertexProperty& vp = (*graphT_)[v];

  return vp.get_is_persistent();
}

//--------------------------------------------------------------------

void ExpressionTree::set_expr_is_persistent( const Tag& tag,
                                             FieldManagerList& fml )
{
  if( !factory_.have_entry(tag) ) return;

  const ExpressionID id = factory_.get_id(tag);

  ID2Vert::iterator ievmt = exprVertexMapT_.find(id);

  if( ievmt == exprVertexMapT_.end() ) return;

  VertexProperty& vpT = (*graphT_)[ievmt->second];

  if( !vpT.get_is_persistent() ){
    FieldDeps& fd = *exprFieldMap_[id];
    vpT.set_is_persistent( true );
    fd.lock_fields( fml );

    // jcs this is dangerous here because it isn't consistent with the behavior
    //     when using VertexProperty::set_is_persistent()
    //     I think that this should really be handled in the scheduler.
    if( hasRegisteredFields_ ){
      if( vpT.execTarget == CPU_INDEX ){
        fd.set_memory_manager(fml, MEM_EXTERNAL, vpT.execTarget );
      }
# ifdef ENABLE_CUDA
      else if ( IS_GPU_INDEX(vpT.execTarget) ){
        fd.set_memory_manager(fml, MEM_STATIC_GPU, vpT.execTarget );
      }
# endif
      else{
        std::ostringstream msg;
        msg << "Invalid Target execution Target : " << vpT.execTarget << " at "
            << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }
    }
  }
}

//--------------------------------------------------------------------

bool
ExpressionTree::computes_field( const Tag& tag ) const
{
  if( !factory_.have_entry(tag) ) return false;
  const ExpressionID id = factory_.get_id( tag );

  // we compute the field if we have the field and an expression to compute it
  if( exprFieldMap_.find(id) == exprFieldMap_.end() ) return false;
  if( !has_expression(id) )                           return false;

  const Vertex& vert = find_vertex( *graph_, id );
  const ExpressionBase* expr = (*graph_)[vert].expr;
  if( expr->is_placeholder() ) return false;

  // if we got here, then we have an expression for the field and it
  // is not a placeholder, so we are computing the field
  return true;
}

//--------------------------------------------------------------------

bool
ExpressionTree::has_expression( const Tag& tag ) const
{
  if( !factory_.have_entry(tag) ) return false;
  return has_expression( factory_.get_id( tag ) );
}

//------------------------------------------------------------------

void
ExpressionTree::prune( TreePtr child )
{
# ifdef PRUNE_DIAGNOSTICS
  cout << " Pruning tree named '" << name() << "' with roots: " << endl;
  BOOST_FOREACH( const ExpressionID id, rootIDs_ ){
    cout << "    " << factory_.get_labels(id) << endl;
  }
  cout << endl;
# endif // PRUNE_DIAGNOSTICS
  IDSet exprRemoval;
  {
    const Graph& childGraph = *(child->graph_);
    const std::pair<VertIter,VertIter> verts = boost::vertices( childGraph );
    for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
      const ExpressionID& eid = childGraph[*iv].id;
      exprRemoval.insert( eid );
    }
  }

  IDSet retainedFields;
  BOOST_FOREACH( const ExpressionID id, rootIDs_ ){
    FieldPruneVisitor fp( retainedFields, exprRemoval );
    boost::breadth_first_search( *graph_, find_vertex(*graph_,id),
                                 boost::color_map( boost::get(&VertexProperty::color,*graph_)).visitor(fp) );
  }
  //
  // remove vertices (expressions) from the parent graph
  // propagate forced persistence characteristics from parent to child graph
  //
  const IDSet& childRoots = child->get_roots();

  BOOST_FOREACH( const ExpressionID id, exprRemoval ){

    // See if this vertex is a root node in the child graph. We don't remove
    // those fields since they will be connected to the parent graph too.
    const bool isChildRoot     =     childRoots.count( id ) > 0;
    const bool isRetainedField = retainedFields.count( id ) > 0;
    const bool keepField = isChildRoot || isRetainedField;

    // find the vertex corresponding to this ID, then prune it
    const Vertex& vert = find_vertex( *graph_, id );
    if( keepField ){
      // replace the existing expression by a placeholder
      ExpressionBase* expr = (*graph_)[vert].expr;

      // jcs note that this will leak memory.
      (*graph_)[vert].expr = expr->as_placeholder_expression();

      // if we have any edges connecting to this vertex in the child graph, mark them as persistent
#     ifdef PRUNE_DIAGNOSTICS
      cout << "  setting persistence on field: " << factory_.get_labels(id) << endl;
#     endif
      if( child->has_expression(id) ){
        (*child->graphT_)[child->find_vertex(*graphT_,id)].set_is_persistent(true);
        (*       graphT_)[       find_vertex(*graphT_,id)].set_is_persistent(true);
        (*child->graphT_)[child->exprVertexMapT_[id]     ].set_is_persistent(true);
        (*       graphT_)[       exprVertexMapT_[id]     ].set_is_persistent(true);
      }
    }
    else{
#     ifdef PRUNE_DIAGNOSTICS
      cout << "  removing expression: " << factory_.get_labels(id) << endl;
      cout << "  removing field: " << factory_.get_labels(id) << endl;
#     endif
      // remove the expression from the graph.
      boost::clear_vertex ( vert, *graph_ );
      boost::remove_vertex( vert, *graph_ );
      exprFieldMap_.erase( id );
    }

  }
# ifdef PRUNE_DIAGNOSTICS
  const std::pair<VertIter,VertIter> verts = boost::vertices( *graph_ );
  cout << endl << "  retained expressions: " << endl;
  for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
    cout << "    " << (*graph_)[*iv].expr->get_tags() << endl;
  }
  cout << endl;
# endif

# ifdef PRUNE_DIAGNOSTICS
  cout << endl << "  retained fields: " << endl;
  for( ExprFieldMap::const_iterator ifld = exprFieldMap_.begin(); ifld!=exprFieldMap_.end(); ++ifld )
    cout << "    " <<  *(ifld->second) << endl;
  cout << endl << " done" << endl;
# endif
  //
  // reset index map on VertexProperty to ensure that it ranges from 0 to #verts
  //
  {
    int ix=0;
    const std::pair<VertIter,VertIter> verts = boost::vertices(*graph_);
    for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
      (*graph_)[*iv].index = ix++;
    }
  }

  delete graphT_;
  graphT_ = new Graph();
  boost::transpose_graph( *graph_, *graphT_,
            boost::vertex_index_map( boost::get(&VertexProperty::index, *graph_) ) );

# ifdef ENABLE_CUDA
  scheduler_.reset( new HybridScheduler(*graph_,*graphT_) );
# else
  scheduler_.reset( new PriorityScheduler(*graph_,*graphT_) );
# endif

  // Rebuild vertex mappings
  exprVertexMapT_.clear();
  std::pair<VertIter, VertIter> vertx = boost::vertices(*graphT_);
  for( VertIter vit = vertx.first; vit != vertx.second; ++vit ){
    exprVertexMapT_.insert( std::make_pair( (*graphT_)[*vit].id, *vit) );
  }
}

//------------------------------------------------------------------

Vertex
ExpressionTree::find_vertex( const Graph& g,
                             const ExpressionID id,
                             const bool require ) const
{
  const std::pair<VertIter,VertIter> verts = boost::vertices(*graph_);
  for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
    if( id == g[*iv].id ) return *iv;
  }

  if( !require ) return Vertex();

  std::ostringstream msg;
  msg << endl << __FILE__ << " : " << __LINE__ << endl
      << "in ExpressionTree::find_vertex()" << endl
      << "no expression with id " << id << " and tag "
      << factory_.get_labels(id) << " was found" << endl
      << "Expressions on this tree follow: ";
  for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
    msg << g[*iv].expr->get_tags()[0] << endl;
  }
  throw std::runtime_error( msg.str() );
}

//------------------------------------------------------------------

//This will transpose the graph, and calculate optimal start and finish times for each node.
bool
ExpressionTree::push_schedule_timings()
{
  Graph gt;

  if( !bTimingsReady_ ) { pScore_ = -1; return false; }

  const std::pair<VertIter,VertIter> verts = boost::vertices(*graphT_);
  // find the end vertices in the original graph.
  tOne_ = 0;
  for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
    VertexProperty& vp = (*graphT_)[*iv];
    tOne_ += (vp.execTarget == CPU_INDEX ) ? vp.vtp.eTimeCPU_ : vp.vtp.eTimeGPU_;

    if( vp.nparents == 0 ){
      vp.vtp.sTime_ = 0;
      vp.vtp.fTime_ = (vp.execTarget == CPU_INDEX ) ? vp.vtp.eTimeCPU_ : vp.vtp.eTimeGPU_;
      boost::breadth_first_search( gt, *iv,
                                   boost::color_map(boost::get(&VertexProperty::color,gt))
                                   .visitor(ParaCalcVisitor(this)) );
    }
  }

  sINF_ = tOne_ / tINF_;

  pScore_ = ( ( 1.0 - ( 1.0 / sINF_ ) ) / ( 1.0 - ( 1.0 / (double)boost::num_vertices(*graphT_) ) ) );

  return true;
}

//------------------------------------------------------------------
#ifdef ENABLE_CUDA
bool
ExpressionTree::is_homogeneous_gpu() const
{
  const Graph& graph = *graphT_;
  const std::pair<VertIter,VertIter> verts = boost::vertices(graph);

  // Uintah Per-patch variables are not ported for GPU yet (Jun7,2014)
  // time stepping variables sometimes gets a placeholder exprs.
  for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
    const VertexProperty& vp = graph[*iv];
    if( (vp.expr->get_tags()[0].name() == "time"     ||
         vp.expr->get_tags()[0].name() == "timestep" ||
         vp.expr->get_tags()[0].name() == "dt"       ||
         vp.expr->get_tags()[0].name() == "rkstage") && vp.expr->is_gpu_runnable() )
      vp.expr->set_gpu_runnable( false );
  }

  // Introspecting GPU runnable property for expressions
  for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
    const VertexProperty& vp = graph[*iv];
    if( !vp.expr->is_gpu_runnable() ) return false;
  }
  return true;
}

//------------------------------------------------------------------

void
ExpressionTree::turn_off_gpu_runnable()
{
  const Graph& graph_ = this->get_graph( true );
  const std::pair<VertIter,VertIter> verts = boost::vertices(graph_);

  for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
    const VertexProperty& vp = graph_[*iv];
    const ExpressionID& exprID = graph_[*iv].id;
    ExpressionBase& expr = get_expression_factory().retrieve_expression( exprID, patchID_ );

    // store the GPU runnable property of all expressions before flipping,
    // so that this property can be restored.
    exprTargetMap_.insert( std::pair< ExpressionID, bool >(vp.id, expr.is_gpu_runnable()) );

    // If the expression is GPU runnable, set it to CPU runnable
    if( expr.is_gpu_runnable() ) expr.set_gpu_runnable( false );
  }
}

//------------------------------------------------------------------

void
ExpressionTree::restore_gpu_runnable()
{
  const Graph& graph_ = *graphT_;
  const std::pair<VertIter,VertIter> verts = boost::vertices(graph_);
  bool gpuBeforeFlip, gpuAfterFlip;
  for( VertIter iv=verts.first; iv!=verts.second; ++iv ){
    const ExpressionID& exprID= graph_[*iv].id;
    const VertexProperty& vp = graph_[*iv];

    gpuBeforeFlip = exprTargetMap_.find(exprID)->second; // property before the flip
    gpuAfterFlip  = vp.expr->is_gpu_runnable();         // property after the flip

    if( exprTargetMap_.find(exprID) != exprTargetMap_.end() ) {
      if( !gpuAfterFlip && gpuBeforeFlip ) vp.expr->set_gpu_runnable(true);
    }
    else{
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "ERROR ! restore_gpu_runnable() for expression : " << vp.expr->get_tags()[0] << std::endl
          << ". turn_off_gpu_runnable() should be called before using this method." << std::endl;
      throw std::runtime_error(msg.str());
    }
  }

  // Change of node hardware targets and field locations has to be informed to the Hybrid scheduler
  scheduler_->invalidate();
  scheduler_->setup( true );
}
# endif // ENABLE_CUDA


} // namespace Expr
