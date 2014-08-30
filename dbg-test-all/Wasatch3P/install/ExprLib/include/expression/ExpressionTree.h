/*
 * \file ExpressionTree.h
 *
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
#ifndef ExpressionTree_h
#define ExpressionTree_h

// Standard
#include <map>
#include <vector>
#include <set>
#include <string>
#include <ostream>
#include <time.h>

// Boost
#include <boost/graph/visitors.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/shared_ptr.hpp>

#include <expression/ExpressionID.h>
#include <expression/FieldDeps.h>
#include <expression/ExpressionBase.h>
#include <expression/GraphType.h>
#include <expression/Schedulers.h>

namespace Expr{

// forward declarations
class ExpressionBase;
class ExpressionFactory;

/**
 *  @class  ExpressionTree
 *  @author James C. Sutherland
 *  @date   May, 2007
 *  @brief  Holds a collection of Expression objects that form a tree.
 *
 *  The ExpressionTree class holds a collection of expressions on a
 *  patch.
 *
 *  To build an ExpressionTree, you must specify the root of the tree.
 *  This is simply an ExressionID, which identifies a unique
 *  expression.
 *
 *  Expressions must be registered with the ExpressionFactory, which
 *  manages creation of expressions.  Then any expression can use
 *  another expression by advertising its ID as a dependent ID.
 *
 *  Calling the <code>compile_expression_tree()</code> method on an
 *  ExpressionTree results parsing of the dependency graph and
 *  retrieval all child expressions from the
 *  ExpressionFactory.
 *
 *  Expression trees may not have circular dependencies.
 *
 *  @par Summary of ExpressionTree Generation
 *
 *   \li Generate a tree by providing the ExpressionID for the root of
 *   the tree.  This expression is then constructed, and is queried
 *   for its dependents via a call to the
 *   <code>Expression::advertise_dependents()</code> method.  The
 *   dependents are recursively interrogated until the whole tree is
 *   built.  As each dependent is discovered, it is retrieved from the
 *   factory.
 *
 *   \li After the tree is compiled, and before the tree is executed,
 *   the patch should allocate fields.  After field allocation on the
 *   patch, the <code>Expression::bind_fields()</code> method should
 *   be called on the ExpressionTree objects.  This will allow all
 *   Expression objects to bind concrete instances of fields.
 *
 *  @todo Need to allow for "post-processing" trees that are separate
 *  and may be executed separately but do not require recalculation of
 *  expressions contained in other trees.  Perhaps we could supply a
 *  main tree and then prune tree execution for expressions contained
 *  in the main tree???
 *
 *  @todo Consider implementing a convenience function to consolidate
 *  calls to register_fields, allocate_fields, bind_fields,
 *  bind_operators.
 */
class ExpressionTree
{
  ExpressionTree();  // no default construction
  ExpressionTree( const ExpressionTree& );  // no copying

  /**
   *  \class ExecMutex
   *  \brief Scoped lock.
   */
    class ExecMutex
    {
#   ifdef ENABLE_THREADS
      const boost::mutex::scoped_lock lock;
      inline boost::mutex& get_mutex() const
      {
        static boost::mutex m;
        return m;
      }
    public:
      ExecMutex() : lock( get_mutex() ){}
      ~ExecMutex(){}
#   else
    public:
      ExecMutex(){}
      ~ExecMutex(){}
#   endif
    };

public:
  typedef std::set<ExpressionID>              RootIDList;

  typedef boost::shared_ptr< ExpressionTree > TreePtr;
  typedef std::vector<TreePtr>                TreeList;

  typedef std::map<ExpressionID,boost::shared_ptr<FieldDeps> > ExprFieldMap;

  //@{

  typedef std::map<ExpressionID,int> ID2Index; ///< Map the ExpressionID to the vertex index number

  typedef RootIDList IDSet;

  //@}

  /**
   *  @brief Construct an expression tree, with the specified
   *  Expression as the root.
   *
   *  @param rootID The ExpressionID for the Expression which will
   *  serve as the root of the tree.
   *  @param factory The ExpressionFactory object used to construct expressions.
   *  @param patchID An identifier that this tree is associated with.  This allows
   *  construction of multiple unique trees (and expressions) associated with each unique patchID.
   *  @param treeName a string identifier for this tree.
   */
  ExpressionTree( const ExpressionID rootID,
                  ExpressionFactory& factory,
                  const int patchID,
                  const std::string treeName = "unnamed" );

  /**
   *  @brief Construct an expression tree, with the specified
   *  Expression as the root.
   *
   *  @param ids The list of Expressions defining the root of the tree.
   *  @param factory The ExpressionFactory object used to construct expressions.
   *  @param patchID An identifier that this tree is associated with.  This allows
   *  construction of multiple unique trees (and expressions) associated with each unique patchID.
   *  @param treeName a string identifier for this tree.
   */
  ExpressionTree( const RootIDList& ids,
                  ExpressionFactory& factory,
                  const int patchID,
                  const std::string treeName = "unnamed" );

  /**
   *  @brief Builds an "empty" ExpressionTree
   *
   *  @param factory The ExpressionFactory object used to construct expressions.
   *  @param patchID An identifier that this tree is associated with.  This allows
   *  construction of multiple unique trees (and expressions) associated with each unique patchID.
   *  @param treeName a string identifier for this tree.
   */
  ExpressionTree( ExpressionFactory& factory,
                  const int patchID,
                  const std::string treeName = "unnamed" );

  virtual ~ExpressionTree();

  /** add another tree to this one. */
  //@{
  void insert_tree( const ExpressionID );
  void insert_tree( const IDSet& );
  void insert_tree( ExpressionTree& );
  //@}

  /**
   *  @brief Build the tree starting from the specified root ID and
   *  working through its dependents recursively.
   *
   *  Upon return, all expressions in this tree have been constructed
   *  and the whole tree has been bound.  The tree may then be
   *  evaluated by calling the
   *  <code>ExpressionTree::execute_tree()</code> method.
   */
  void compile_expression_tree();

  /**
   *  @brief Register all fields associated with this ExpressionTree.
   *  This should be done after any tree splitting is completed.
   */
  void register_fields( FieldManagerList& );   // jcs deprecate use?

  /**
   * @brief Register all fields associated with this ExpressionTree using a
   *  collection of FieldManagerLists to allow different fields to potentially
   *  be managed on different meshes.
   */
  void register_fields( FMLMap& );

  /**
   * @brief Lock all fields associated with this ExpressionTree.
   * Note: This will disable freeing of any non-persistent fields
   */
  void lock_fields( FieldManagerList& );
  void lock_fields( FMLMap& );

  /**
    * @brief Unlock all fields associated with this ExpressionTree.
    * Note: This will allow freeing of any non-persistent fields
    */
  void unlock_fields( FieldManagerList& );
  void unlock_fields( FMLMap& );

  /**
   *  @brief Directs all expressions in this tree to obtain field
   *  references from the appropriate FieldManager.
   *
   *  After fields have been allocated, this method should be called
   *  on the tree to allow all expressions to bind fields as needed.
   *  This MUST be called prior to executing the tree.
   */
  void bind_fields( FieldManagerList& );

  /**
   *  @brief Directs all expressions in this tree to obtain field
   *  references from the appropriate FieldManager.
   *
   *  After fields have been allocated, this method should be called
   *  on the tree to allow all expressions to bind fields as needed.
   *  This MUST be called prior to executing the tree.
   */
  void bind_fields( FMLMap& );

  /**
   *  @brief Directs all expressions in this tree to bind operators as necessary.
   *
   *  This method may need to be called during the course of time
   *  integration in cases where the mesh is changing in time.
   */
  void bind_operators( const SpatialOps::OperatorDatabase& );

  /**
   *  @brief Directs all expressions in this tree to bind operators as necessary.
   *
   *  This method may need to be called during the course of time
   *  integration in cases where the mesh is changing in time.
   */
  void bind_operators( const OpDBMap& );

  /**
   *  @brief Execute all expressions in the tree.
   *
   *  The tree is traversed and each expression is executed, beginning
   *  from the one at the "bottom" of the tree.
   *
   *  @todo This could be implemented in a threaded approach, but we
   *  will need to ensure thread safety of all of the code first...
   */
  void execute_tree();

  /**
   *  @brief Each tree is composed of one or more roots.  Return the
   *  entire vector of roots for this tree.
   */
  const RootIDList& get_roots() const { return rootIDs_; }

  /**
   *  @brief Query if we have the given expression in this tree.
   */
  bool has_expression( const ExpressionID& id ) const;
  bool has_expression( const Tag& tag ) const;

  /**
   *  \brief query if a given field is present in the tree.  Note that
   *  for cleaved trees, a field may be present even if it is not
   *  computed (associated with an expression).
  */
  bool has_field( const Tag& tag ) const;

  /**
   * \brief sets the specified field as persistent, meaning it is not eligible
   * for dynamic memory allocation
   */
  void set_expr_is_persistent( const Tag& tag , FieldManagerList& fml);

  /**
   * \brief return whether or the specified field is tagged as persistent.
   *  Note: this assumes the field exists. If it does not, the return value
   *  will be false.
   */
  bool is_persistent( const Tag& tag ) const;

  /**
   *  \brief query if a field is present in the tree and has an
   *  expression that computes it.
   */
  bool computes_field( const Tag& tag ) const;

  /** @brief Obtain the ID for an expression with the given label.
   *  This queries the registry, not only members of this tree.
   */
  ExpressionID get_id( const Tag& label ) const;

  /** @brief obtain the name for this ExpressionTree */
  std::string name() const{ return name_; }

  ExpressionFactory& get_expression_factory() { return factory_; }

  /** @brief Write the tree to a format readable by GraphViz.  See http://www.graphviz.org
   *  @param os the output stream
   *  @param execTree [optional] if true, then the execution graph will be output.
   *  Default (true) outputs the dependency graph.
   *  @param details [optional] if true, then modifier expressions will also be output.
   */
  void write_tree( std::ostream& os,
                   const bool execTree=false,
                   const bool details=false ) const;

  /**
   *  @brief Split the tree into a collection of trees.  Splitting
   *  occurs where expressions have been tagged as requiring a ghost
   *  update.
   */
  TreeList split_tree();

  bool operator==( const ExpressionTree& other ) const;

  /** \brief obtain the map of fields used in this ExpressionTree */
  const ExprFieldMap& field_map() const{ return exprFieldMap_; }

  /** \brief Update parallelization and speedup scores.
   *     Calling this function will cause the expression tree to compute
   *     an estimate for the theoretic parallelizability (p score) and
   *     speedup (s score) for its task graph. This requires a subsequent
   *     call to execute before scores will be available.
   *
   *   NOTE:
   *     This is only an estimate and will vary somewhat, depending
   *     on the underlying software platform and hardware utilization.
   */
  void update_graph_scores()
  {
    if( !push_schedule_timings() ){ pScore_ = 0; sINF_ = 0; bUpdatePScore_ = true; }
  }

  /** \brief Get Parallelization Score
   *    Theoretic maximum fraction of the graph which can be computed in parallel.
   *
   *    Returns 0 if no score has been calculated.
   */
  double get_p_score() const{ return this->pScore_; }

  /** \brief Get Speedup Score
   *    Theoretic maximum speedup that can be acheived with the current graph.
   *
   *    Returns 0 if no score has been calculated.
   */
  double get_s_score() const{ return this->sINF_; }

  /**
   *  \brief given the graph and an ExpressionID, this returns the associated graph vertex
   */
  Vertex find_vertex( const Graph& graph, const ExpressionID, bool require=true ) const;

  /**
   * @param execGraph [optional] if true, the execution graph is returned.
   *   Default (false) returns the dependency graph.
   * @return the underlying boost::graph object
   */
  const Graph& get_graph(const bool execGraph=false) const{ return (execGraph ? *graphT_ : *graph_); }

  int patch_id() const{ return patchID_; }

  bool is_cleaved() const{ return isCleaved_; }

  /**
   * \brief Write out the vertex properties for the nodes in the graph.  Used only for diagnostics/debugging.
   * \param os the output stream
   * \param execGraph optional.  If true, then the execution graph will be dumped.  Default (false) is the dependency graph.
   */
  void dump_vertex_properties( std::ostream& os, const bool execGraph=false ) const;


# ifdef ENABLE_CUDA

  /**
   * \brief block the host thread until the works assigned to the cuda Stream is complete.
   *
   *  Note - This method is thread-safe
   */
  void wait_on_cuda_stream();

  /**
   * \brief sets the device index to tree, scheduler and it's components.
   *        If the device context is already assigned, these are the following implications
   *
   *        1. Memory associated with fields are deallocated for the older device index
   *        2. Scheduler is invalidated so that it is setup again
   *
   * \param deviceIndex to set on the tree
   *
   * \param FieldManagerList passed for allocating and deallocating fields which
   *        happens for the case when device context changes
   */
  void set_device_index( const unsigned int deviceIndex, FieldManagerList& fml );

  /**
   * \brief returns the device index of tree
   */
  int get_device_index() const;

  /**
   * \brief determines if all nodes in the graph are GPU runnable.
   */
  bool is_homogeneous_gpu() const;

  /**
   * \brief Turns off the GPU runnable property for the expressions.
   */
  void turn_off_gpu_runnable();

  /**
   * \brief restores the gpu runnable property for the expressions that have
   *        been turned off by turn_off_gpu_runnable()
   *
   * Note : calling restore_gpu_runnable() before turn_off_gpu_runnable()
   *        results in a exception
   */
  void restore_gpu_runnable();
# endif

protected:

  const std::string name_;
  int patchID_;

  RootIDList rootIDs_;

  ExpressionFactory& factory_;

  bool hasRegisteredFields_;
  bool hasBoundFields_;
  bool bUpdatePScore_, bTimingsReady_;

  double pScore_, tOne_, tINF_, sINF_; //parallelization score and data read time

  /** Boost graph related things */

  // define a directed graph with only out-edge traversal.  We could
  // get bidirectional traversal easily if needed...
  typedef boost::graph_traits<Graph>::edge_iterator     EdgeIter; ///< Edge iterator

  typedef std::map<ExpressionID, bool> EXPR2TARGET;
  typedef std::map<ExpressionID, Vertex> ID2Vert;

  // pruning stuff:
  typedef std::set   <Vertex>       VertList;
  typedef std::vector<Edge>         EdgeList;

  //@}

  //Scheduler typedefs
  typedef boost::shared_ptr< Scheduler > TaskScheduler;

  /**
   *  @brief The boost::graph object used to describe the dependency tree.
   */
  Graph* graph_;
  Graph* graphT_;
  EXPR2TARGET exprTargetMap_;
  ID2Vert exprVertexMapT_;    ///< for internal use only.
  mutable bool isCleaved_;
  ExprFieldMap exprFieldMap_;
  TaskScheduler scheduler_;
  PollerList pollers_;
  NonBlockingPollerList nonBlockPollers_;

# ifdef ENABLE_CUDA
  unsigned int deviceID_;    ///< device ID set to Expression Tree
  bool deviceAlreadySet_;    ///< info regarding whether the device is already set or not
# endif

  //------------------------------------------------------------------

  struct ParaCalcVisitor : public boost::default_bfs_visitor
  {
    ParaCalcVisitor( ExpressionTree* parent ) {
      parent_ = parent;
    }
    ExpressionTree* parent_;

    void examine_edge( Edge e, const Graph& g )
    {
      const Vertex src = boost::source(e,g);
      const Vertex dst = boost::target(e,g);

      Graph& g2 = const_cast<Graph&>(g);

      g2[dst].vtp.sTime_ = g2[dst].vtp.sTime_ >= g2[src].vtp.fTime_ ? g2[dst].vtp.sTime_ : g2[src].vtp.fTime_;
      g2[dst].vtp.fTime_ = g2[dst].vtp.sTime_ + ( g2[dst].execTarget == CPU_INDEX ? g2[dst].vtp.eTimeCPU_ : g2[dst].vtp.eTimeGPU_ );

      parent_->tINF_ = parent_->tINF_ >= g2[dst].vtp.fTime_ ? parent_->tINF_ : g2[dst].vtp.fTime_;
    }
  };

  //------------------------------------------------------------------

  void prune( TreePtr child );
  //}@

  /**
   *  Recursively descend through the expressions in this tree to
   *  determine the full dependency tree.  As this is done, a list of
   *  edges is created to be used in generating the graph.
   *
   * @param[in] id
   * @param[out] vertIx The index for the vertex.  This must be zero-based
   *         whereas the ExpressionID might not be.  If added, this
   *         will be incremented.
   * @param[out] id2index The map that returns the appropriate vertex
   *        identifier given the ExpressionID
   * @param[out] depth how deep from the root nodes this expression is.
   */
  void bootstrap_dependents( const ExpressionID& id,
                             int& vertIx,
                             ID2Index& id2index,
                             int& depth );

  /**
   *  If the given vertex does not appear in the graph, it will be added.
   *
   *  @param[out] v The vertex in the graph corresponding to the supplied expression.
   *
   *  @param[out] alreadyPresent true if the id was already associated with a vertex.
   *
   *  @param[out] vertIx The index for the vertex.  This must be zero-based
   *         whereas the ExpressionID might not be.  If added, this
   *         will be incremented.
   *
   *  @param[out] id2index The map that returns the appropriate vertex
   *         identifier given the ExpressionID
   *
   *  @param[in] id The ExpressionID for the Expression to be added to
   *         this vertex in the graph.
   */
  void add_vertex_to_graph( Vertex& v,
                            bool& alreadyPresent,
                            int& vertIx,
                            ID2Index& id2index,
                            const ExpressionID& id );

  /** \brief Determine start and finish times given a graph with known execution timings. **/
  bool push_schedule_timings();

  friend void cleave_tree( TreePtr parent, IDSet& taggedList, TreeList& treeList );
};


//====================================================================



} // namespace Expr


#endif
