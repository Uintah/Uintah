/**
 * \file ExpressionBase.h
 * \author James C. Sutherland
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
#ifndef ExpressionBase_h
#define ExpressionBase_h

#include <set>
#include <cassert>

#include <expression/ExpressionID.h>
#include <expression/Tag.h>
#include <expression/SourceExprOp.h>
#include <expression/FieldManagerList.h>
#include <expression/ExprDeps.h>
#include <expression/FieldDeps.h>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace SpatialOps{ class OperatorDatabase; }

namespace Expr{

  class ExpressionBase;

  /**
   *  @class  ExpressionBuilder
   *  @author James C. Sutherland
   *  @date   June, 2007
   *
   *  @brief Base class for use in building expressions.  Interfaces to
   *  the ExpressionFactory.
   */
  class ExpressionBuilder
  {
  private:
    TagList computedFields_;
    int fmlID_;

  public:

    /**
     * @brief Construct an ExpressionBuilder for an Expression that computes multiple fields
     */
    ExpressionBuilder( const TagList& computedFields ){ computedFields_ = computedFields; }

    /**
     * @brief Construct an ExpressionBuilder for an Expression that computes one field
     */
    ExpressionBuilder( const Tag& tag ){ computedFields_.push_back(tag); }

    virtual ~ExpressionBuilder(){};

    /**
     *  This method is called to construct the Expression. It should
     *  build the Expression using the \c new command so that it is
     *  allocated on the heap.  Ownership of the object is transferred
     *  to the ExpressionFactory.
     */
    virtual ExpressionBase* build() const = 0;

    /**
     * @brief Obtain the TagList for the fields computed by the expression.
     */
    const TagList& get_tags() const{ return computedFields_; }
  };


  //====================================================================


  /**
   *  @class  ExpressionBase
   *  @date   June, 2007
   *  @author James C. Sutherland
   *
   *  @brief Abstract base class for all Expression classes.
   *
   *  This class is not templated so that we can have containers of
   *  ExpressionBase objects.  Expression classes should not generally
   *  derive directly from this class.  Rather, they should derive from
   *  the Expression class, which provides additional type information.
   *
   *  See documentation of the Expression class for more information.
   */
  class ExpressionBase
  {
  public:

    /** \brief set the expression as GPU runnable */
    void set_gpu_runnable( const bool b );

    /** \brief returns Expression's GPU eligibility flag */
    inline bool is_gpu_runnable() const{ return exprGpuRunnable_; }

    /**
     *  @brief Construct an ExpressionBase object.
     */
    ExpressionBase();

    virtual ~ExpressionBase();

    /**
     *  @brief Set the Tag(s) that this expression computes.
     *         Typically only called internally by the Expression class.
     */
    void set_computed_tag( const TagList& tags );

    /** @brief obtain the Tag (name) for this expression */
    const Tag& get_tag() const;

    /** @brief obtain the Tags (names) computed by this expression */
    const TagList& get_tags() const{ return exprNames_; }

    /**
     *  \brief returns true if this expression is a placeholder
     *  (i.e. does not compute anything, just serves to wrap a field).
     */
    virtual bool is_placeholder() const{ return false; }

    /**
     *  @brief Add an expression to this one that should be subtracted from it.
     *
     *  This is particularly useful to add MMS source terms to RHS
     *  expressions.  In general, it should not be used.
     *
     *  \param src The expression to attach to this one
     *
     *  \param op The operation to be performed to augment this expression
     *     by the "src" expression.
     *
     *  \param myFieldIx For expressions computing multiple fields, this
     *    must be specified to indicate which field should be augmented by
     *    the source term expression.
     *
     *  \param srcFieldIx For src expressions computing multiple fields, this
     *    must be specified to indicate which field should be used to augment
     *    this expression value.
     */
    void attach_source_expression( const ExpressionBase* const src,
                                   const SourceExprOp op,
                                   const int myFieldIx = 0,
                                   const int srcFieldIx = 0 );

    /** \brief query if this expression is to be cleaved from its parents */
    inline bool cleave_from_parents() const{ return cleaveFromParents_; }

    /** \brief query if this expression is to be cleaved from its children */
    inline bool cleave_from_children() const{ return cleaveFromChildren_; }

    /**
     *  \brief Call this method on the expression to indicate that
     *  this expression should be cleaved from its parents when
     *  constructing the tree.
     */
    inline void cleave_from_parents( const bool r ){ cleaveFromParents_=r; }

    /**
     *  \brief Call this method on the expression to indicate that
     *  this expression should be cleaved from its children when
     *  constructing the tree.
     */
    inline void cleave_from_children( const bool r ){ cleaveFromChildren_=r; }

    /**
     * \brief in situations where more than one FieldManagerList is present
     *  (e.g. multiple patches in a single graph), this provides a mechanism
     *  to associate this expression with the appropriate FieldManagerList.
     * \param fmlid the identifier for the FieldManagerList
     */
    inline void set_field_manager_list_id( const int fmlid ){ fmlID_=fmlid; }

    inline int field_manager_list_id() const{ return fmlID_; }

    /**
     *  \brief Expose dependencies and store them in the supplied
     *   objects. Only intended for internal use by the ExpressionTree.
     *
     *   \param exprDeps expressions that are required by this expression.
     *   \param fieldDeps fields computed by this expression, populated
     *     by the set_computed_fields method.
     */
    void base_advertise_dependents( ExprDeps& exprDeps, FieldDeps& fieldDeps);

    /**
     *  \brief Bind fields used by this Expression. Only intended for
     *   internal use by the ExpressionTree.
     */
    void base_bind_fields( FieldManagerList& fldMgrList );
    void base_bind_fields( FMLMap& fmls );

    /**
     *  \brief Bind operators used by this Expression. Only intended
     *   for internal use by the ExpressionTree.
     */
    void base_bind_operators( const SpatialOps::OperatorDatabase& opDB );

    // virtual and implemented in Expression class since we need type information.
    virtual void base_evaluate() = 0;

    /**
     * \brief Add a modifier expression to this one.  This method should only be
     *  accessed by the ExpressionFactory in general.
     *
     *  Modifier expressions provide a mechanism to set boundary conditions on
     *  fields and introduce arbitrary additional dependencies to do so.  This
     *  is needed to impose non-trivial boundary conditions and ensures proper
     *  ordering of the resulting expression graph.  The modifier expression
     *  is not a "first class" expression in that it is not directly represented
     *  on the graph. Rather, it influences the graph by piggy-backing on an
     *  expression, where its dependencies are incorporated.  Modifiers are
     *  executed immediately after an expression completes.
     *
     * \param modifier the expression to modify this one.
     * \param modTag the original tag used to identify this modifier. Because the
     *   modifier "evaluates" the same field as this expression that it is
     *   attached to, the modTag allows us to remember the name originally
     *   associated with the expression.
     */
    void add_modifier( ExpressionBase* const modifier, const Tag& modTag );

    /**
     * \brief Obtain the typeid name for the field type that this expression computes.
     *
     * This is primarily used internally for type consistency checking when only
     * base class objects are available.
     *
     * @return the typeid name for the field type associated with this expression
     */
    virtual const char* field_typeid() const = 0;

    /**
     * Returns the number of "post-processing" functors or expressions hanging
     * on this expression. These are attached via calls to \c process_after_evaluate()
     * or \c ExpressionFactory::attach_modifier().
     */
    virtual int num_post_processors() const = 0;

    /**
     * @return the names of all of the functors and modifiers on this expression.
     * See also num_post_processors().
     */
    virtual std::vector<std::string> post_proc_names() const =0;

    /**
     * Allows conversion of this expression into a PlaceHolder.  This should only be used for cleaving.
     */
    virtual ExpressionBase* as_placeholder_expression() const =0;

#   ifdef ENABLE_CUDA

    /**
     * \brief creates a cuda stream for this expression on a specified device.
     *
     * @param deviceIndex which provides device context for creating streams
     *
     * Note : If the stream already exists and there is a change in device context,
     *        the previous existing stream is destroyed and new stream is created for the device context.
     */
    void create_cuda_stream( int deviceIndex );

    /**
     * \brief returns the cudaStream set on the expression
     */
    cudaStream_t get_cuda_stream();

#   endif

  protected:

    /**
     *  @brief Evaluate this expression and cache the result.
     *
     *  The <code>evaluate()</code> method should perform the following:
     *
     *   \li Obtain the values of expressions that it depends on via the
     *   <code>Expression::value()</code> method.
     *
     *   \li Calculate the value of this expression and store it in the
     *   variable returned by <code>Expression::value()</code> for
     *   single variable expressions and
     *   <code>Expression::get_value_vec()</code> for multi-value
     *   expressions.
     */
    virtual void evaluate() = 0;


    /**
     *  @brief Advertise the expressions that this expression directly
     *  depends on.
     *
     *  Derived classes should use this method to specify the
     *  Expressions that they depend on.  This is done by calling one of
     *    \code
     *       ExprDeps::register_expression( const Tag& )
     *       ExprDeps::register_expression( const ExpressionID& )
     *    \endcode
     */
    virtual void advertise_dependents( ExprDeps& exprDeps ) = 0;


    /**
     *  @brief Obtain pointers to fields as necessary.
     *
     *  Derived classes may specialize this method to bind fields as
     *  needed. A const FieldManagerList is provided to force
     *  Expressions to hold const pointers to fields. Only the field
     *  that an Expression computes should be non-const, and this is
     *  automatically bound (see get_value_vec() or value()).
     */
    virtual void bind_fields( const FieldManagerList& fldMgrList ) = 0;

    /**
     *  @brief Obtain pointers to spatial operators as necessary.
     *
     *  Derived classes should implement this method to obtain pointers
     *  to spatial operators as needed.  In the case of mesh adaptivity,
     *  this method may be called during execution so that operators can
     *  be re-bound.
     *
     *  This method has a default implementation which does nothing. It
     *  is not pure virtual since many expressions do not require any
     *  operators and we don't want to add unneeded complexity in that
     *  case.
     */
    virtual void bind_operators( const SpatialOps::OperatorDatabase& opDB ){}

    struct SrcTermInfo{
      SrcTermInfo( const ExpressionBase* const expr, const SourceExprOp _op, const int thisIx, const int srcTermIx )
        : srcExpr( expr ), op(_op), myIx( thisIx ), srcIx( srcTermIx )
      {}
      const ExpressionBase* const srcExpr;
      const SourceExprOp op;
      const int myIx, srcIx;
    };
    struct SrcTermInfoCompare{
      bool operator()( const SrcTermInfo& s1, const SrcTermInfo& s2 ){
        bool isLess = s1.srcExpr->get_tags()[0].id() < s2.srcExpr->get_tags()[0].id();
        if( s1.srcExpr->get_tags()[0].id() == s2.srcExpr->get_tags()[0].id() ){
          isLess = s1.myIx < s2.myIx;
          if( s1.myIx == s2.myIx ){
            isLess = s1.srcIx < s2.srcIx;
          }
        }
        return isLess;
      }
    };
    typedef std::set<SrcTermInfo,SrcTermInfoCompare> ExtraSrcTerms;
    ExtraSrcTerms srcTerms_;

    /**
     *  \brief Set the field(s) that this expression computes. The
     *    default implementation should be sufficient for most
     *    situations, but this may be re-implemented in derived
     *    classes to customize its behavior.
     *
     *  This method is called from base_advertise_dependents() and is
     *  hidden from user implemented expressions.
     *
     *  If this method is specialized in an expression, the programmer
     *  should also specialize bind_computed_fields().
     */
    virtual void set_computed_fields( FieldDeps& ) = 0;

    /**
     * \brief  Bind the field(s) that this expression computes.
     *
     *  This method is called from base_bind_fields(), and is hidden
     *  from user implemented expressions.  It automatically hooks up
     *  the field(s) that a user-defined expression computes.
     *
     *  If the user wants to over-ride the default behavior, this can
     *  be specialized in the user defined Expression.  However, in
     *  that case, set_computed_fields() should also be specialized.
     */
    virtual void bind_computed_fields( FieldManagerList& ) = 0;

    std::vector<ExpressionBase*> modifiers_;
    std::vector<Tag> modifierNames_;

#   ifdef ENABLE_CUDA
    cudaStream_t cudaStream_;      ///< cuda stream set on thiss expression
    bool cudaStreamAlreadyExists_; ///< Is true, when a stream exists for the expression
    int deviceID_;                 ///< GPU device ID set for expressions
#   endif

  private:

    ExpressionBase( const ExpressionBase& );          ///< no copying
    ExpressionBase& operator=(const ExpressionBase&); ///< no assignment

    bool cleaveFromParents_, cleaveFromChildren_;
    bool exprGpuRunnable_;                            ///< Is true, this expression can run on GPU

    TagList exprNames_;
    int fmlID_;

  };

} // namespace Expr

#endif // ExpressionBase_h
