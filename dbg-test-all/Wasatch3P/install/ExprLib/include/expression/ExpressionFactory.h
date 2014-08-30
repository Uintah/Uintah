/**
 * \file ExpressionFactory.h
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
#ifndef ExpressionFactory_h
#define ExpressionFactory_h

#include <map>
#include <set>
#include <list>

#include <expression/ExprFwd.h>
#include <expression/ExpressionID.h>
#include <expression/Tag.h>
#include <expression/SourceExprOp.h>

namespace Expr{

#define ALL_PATCHES -999999

class ExpressionRegistry; // forward declaration

//====================================================================
/**
 *  @class  ExpressionFactory
 *  @author James C. Sutherland
 *  @date   May, 2007
 *
 *  @brief  Factory to manage creation of Expression objects
 *
 *  The ExpressionFactory class manages creation and
 *  identification of Expression objects.  Any expression
 *  that may potentially come into existance should register itself
 *  here by providing an ExpressionBuilder object to create itself.
 *
 *  Expressions are provided three arguments at construction:
 *
 *   \li ExpressionID - the unique identifier for the expression
 *
 *   \li ExpressionRegistry - information about all
 *   expressions registered.  Expressions may use this to obtain the
 *   ID for expressions they depend on.
 *
 *  @todo Employ smart pointers to eliminate need for memory
 *  management of builders.
 *
 *  @todo Allow a listener pattern here so that if registration
 *  occurs, trees can be recompiled.  This would facilitate dynamic
 *  model transition.
 */
class ExpressionFactory
{
public:

  /**
   *  @brief Construct an ExpressionFactory.
   *
   *  In general, this should be built on each patch.  This provides
   *  separate expressions for each patch.
   *
   *  \param log [false] If true, a log will be written containing
   *  names of expressions registered.
   */
  ExpressionFactory( const bool log = false );

  ~ExpressionFactory();

  /**
   * @brief Call this method to ensure that calls to register_expression() provide a patchID.
   */
  void require_patch_id_specification();

  /**
   *  Any expression that may potentially be used in a simulation must
   *  be registered by name.  No two expressions may be identified by
   *  the same name.  Registration pairs a unique id with the name,
   *  which will remain a unique pairing for the duration of the
   *  simulation.
   *
   *  The factory assumes ownership of the ExpressionBuilder, and will
   *  delete it when it is no longer needed.  Therefore, it should be
   *  heap allocated.
   *
   *  @param builder The ExpressionBuilder that will construct
   *  instances of this Expression. All instances should be allocated
   *  via the \c new operator, and ownership is transfered to the
   *  ExpressionFactory.
   *
   *  @param allowOverWrite If true, then if an expression with a
   *  duplicate same name is added to the registry, it will replace
   *  the existing expression.  If false, then if a duplicate is
   *  added, an exception will occur.  Default is false.
   *
   */
  ExpressionID register_expression( const ExpressionBuilder* builder,
				    const bool allowOverWrite = false );

  /**
   *  @brief this is an advanced interface
   *
   *  @param builder The ExpressionBuilder that will construct
   *  instances of this Expression. All instances should be allocated
   *  via the \c new operator, and ownership is transfered to the
   *  ExpressionFactory.
   *
   *  @param allowOverWrite If true, then if an expression with a
   *  duplicate same name is added to the registry, it will replace
   *  the existing expression.  If false, then if a duplicate is
   *  added, an exception will occur.  Default is false.
   *
   *  @param fmlID the identifier for the FieldManagerList that this expression
   *  should be associated with.  This is useful for situations where multiple
   *  FieldManagerList objects are being used.
   */
  ExpressionID register_expression( const ExpressionBuilder* builder,
                                    const bool allowOverWrite,
                                    const int fmlID );

  /**
   *  Attaches a dependent expression to the specified expression.
   *  This allows one to add functionality to an expression without
   *  modifying its source code.  The dependency expression's value
   *  will be added to the target expression's value.
   *
   *  @param srcTermTag The ExpressionID for the dependency expression.
   *  @param targetTag  The ExpressionID for the expression to attach the dependency to.
   *  @param op add/subtract the srcTermTag to the targetTag.  This controls
   *  whether we add or subtract.
   */
  void attach_dependency_to_expression( const Tag& srcTermTag,
					const Tag& targetTag,
                                        const SourceExprOp op = ADD_SOURCE_EXPRESSION );

  /**
   * Attach a modifier expression.  This expression will be triggered
   * immediately after the value it is intended to modify, but can introduce its
   * own dependencies.  This can be used to achieve boundary conditions on a
   * field, for example, where the BC value is a function of other quantities on
   * the graph.
   *
   * Modifier expressions are not "normal" expressions in the sense that they do
   * not explicitly occupy a position in the graph.  Rather, they are attached
   * to an expression in the graph.
   *
   * @param modifierTag the tag identifying the modifier expression.
   * @param targetTag the tag identifying the expression to be modified.
   * @param patchID the ID for the patch that this modifier should be active on.
   *                If nothing is specified, it will be active on all patches.
   * @param allowOverWrite if true, then if a modifier with the given tag has
   *          already been attached to the given target, it will be overwritten.
   *          If false, then duplicates will result in an exception being thrown.
   */
  void attach_modifier_expression( const Tag& modifierTag,
                                   const Tag& targetTag,
                                   const int patchID=ALL_PATCHES,
                                   const bool allowOverWrite=false );

  void cleave_from_parents( const ExpressionID& id );
  void cleave_from_children( const ExpressionID& id );

  /**
   * @brief get a Poller associated with the requested Tag
   * @param tag the Tag of the expression where the poller is desired.
   */
  PollerPtr get_poller( const Tag& tag );

  const PollerList& get_pollers() const{ return pollerList_; }

  /**
   * @brief obtain the non-blocking poller pointer associated with the associated Tag.
   */
  NonBlockingPollerPtr get_nonblocking_poller( const Tag& tag );
  const NonBlockingPollerList& get_nonblocking_pollers() const{ return nonBlockingPollers_; }

  /** @brief Determine if an expression with the given ExpressionID
      has been registered. */
  bool query_expression( const ExpressionID& id ) const;

  void dump_expressions( std::ostream& os ) const;

  const Tag& get_label( const ExpressionID& ) const;

  TagList get_labels( const ExpressionID& ) const;

  ExpressionID get_id( const Tag& ) const;

  bool have_entry( const Tag& ) const;

  /**
   * @brief for expressions that have a specific FieldManagerList associated with them, this returns the ID.
   *  @param patchID the patchID that the expression is associated with.
   * @param tag the Tag for the expression of interest
   * @return a pair with the first entry indicating if the expression has a FieldManagerList associated with it and the second providing thd ID (if valid)
   */
  std::pair<bool,int> get_associated_fml_id( const int patchID, const Tag& tag ) const;

  /**
   *  Eliminate the expression with the given ExpressionID from the
   *  registry.  Note that this can be very dangerous to do, since it
   *  could wipe out memory being used elsewhere...
   */
  bool remove_expression( const ExpressionID & id );


  /**
   *  @brief Obtain an expression from the factory.  Constructs one if needed.
   *
   *  @param id the ExpressionID for the expression we want to construct
   *
   *  @param patchID the patchID that this expression will be
   *         associated with.  A unique expression will be constructed
   *         and returned for each unique patchID.
   *
   *  @param mustExist If true, then an attempt to retrieve an
   *         expression that has not been built will result in an
   *         exception being thrown.  If false (default), then if the
   *         expression will be created if it does not exist.
   *
   *  This method requires that an expression has been registered.  If
   *  none has been registered, an exception will result.
   *
   *  Needs to have patch information to ensure that we build a unique
   *  expression for each patch.
   *
   *  If no expression with the given ExpressionID has been registered
   *  with the factory, an exception will be thrown.
   */
  ExpressionBase& retrieve_expression( const ExpressionID& id,
                                       const int patchID,
                                       const bool mustExist=false );

  ExpressionBase& retrieve_expression( const Tag& tag,
                                       const int patchID,
                                       const bool mustExist=false );

  /**
   * @brief Retrieve an expression that is known to be a modifier expression.
   *        This is an "advanced" feature and should not be used unless you are
   *        confident that the expression is, indeed, a modifier.
   * @param tag
   *
   * @param patchID the patchID that this expression will be
   *        associated with.  A unique expression will be constructed
   *        and returned for each unique patchID.
   *
   * @param mustExist If true, then an attempt to retrieve an
   *        expression that has not been built will result in an
   *        exception being thrown.  If false (default), then if the
   *        expression will be created if it does not exist.
   *
   * @return The requested modifier expression.
   */
  ExpressionBase& retrieve_modifier_expression( const Tag& tag,
                                                const int patchID,
                                                const bool mustExist=false );

  inline bool is_logging_active() const{ return outputLog_; }

private:

  ExpressionFactory( const ExpressionFactory& );         ///< no copying
  ExpressionFactory& operator=(const ExpressionFactory&); ///< no assignment

  ExpressionBase& retrieve_internal( const ExpressionID& id,
                                     const int patchID,
                                     const bool mustExist,
                                     const bool isModifier );

  /**
   * @brief select the FieldManagerList to be used by this expression.
   *   Only has an effect in situations where more than one FieldManagerList
   *   is available.
   * @param exprID the expression to set the FieldManagerList for.
   * @param listID the integer identifier for the FieldManagerList
   */
  void set_field_manager_list_id( const ExpressionID& exprID, const int listID );

  struct DepInfo{
    ExpressionID srcID;
    SourceExprOp op;
    int targFieldIndex;
    int srcTermFieldIndex;
  };

  struct DepCompare
  {
    bool operator()( const DepInfo& id1, const DepInfo& id2 ){
      if( id1.srcID == id2.srcID ){
        if( id1.op == id2.op ){
          if( id1.targFieldIndex == id2.targFieldIndex ){
            return id1.srcTermFieldIndex < id2.srcTermFieldIndex;
          }
          return id1.targFieldIndex < id2.targFieldIndex;
        }
        else{
          return id1.op < id2.op;
        }
      }
      return id1.srcID < id2.srcID;
    }
  };

  const bool outputLog_;
  bool patchIDRequired_;  // flipped on to require patch ID spec in registration
  bool didSetPatchID_;

  typedef std::set<DepInfo,DepCompare> IDSet;
  typedef std::set<ExpressionID      > CleaveSet;

  typedef std::map<ExpressionID, ExpressionBase*         > IDExprMap;
  typedef std::map<int,          IDExprMap               > PatchExprMap;
  typedef std::map<int,          TagList                 > IDTagListMap;
  typedef std::map<ExpressionID, const ExpressionBuilder*> CallBackMap;
  typedef std::map<ExpressionID, IDSet                   > IDSetMap;
  typedef std::map<ExpressionID, int                     > IDFMLMap;
  typedef std::map<Tag,          IDTagListMap            > ModifierMap;

  CallBackMap callBacks_;
  PatchExprMap exprMap_;
  IDSetMap idSetMap_;
  IDFMLMap idFMLMap_;
  CleaveSet cleaveFromParents_, cleaveFromChildren_;
  PollerList pollerList_;
  NonBlockingPollerList nonBlockingPollers_;
  ModifierMap modifiers_;

  ExpressionRegistry* const registry_;
};

} // namespace Expr

#endif
