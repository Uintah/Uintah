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
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <algorithm>

#include <expression/ExpressionBase.h>
#include <expression/ExpressionFactory.h>
#include <expression/Poller.h>

#include <boost/foreach.hpp>

namespace Expr{


  /**
   *  @class  ExpressionRegistry
   *  @author James C. Sutherland
   *  @date   May, 2007
   *
   *  @brief Registry for <code>Expression</code> objects.
   *
   *  The ExpressionRegistry should not be created.  Rather, it should
   *  only be accessed through the ExpressionFactory.  It is used to
   *  hold a list of Expressions that have been registered.  It does not
   *  provide functionality to construct new expressions or return
   *  references to existing expressions.  The ExpressionFactory should
   *  be used for that purpose.
   *
   *  @todo Ensure that ExpressionRegistry can only be created by the
   *        ExpressionFactory.  Perhaps these two classes should be
   *        combined?
   */
  class ExpressionRegistry
  {
  public:

    /**
     * @brief Obtain the <code>ExpressionID</code> for the expression
     * with the given <code>Tag</code>.
     */
    ExpressionID get_id( const Tag& label ) const;

    const Tag& get_label( const ExpressionID& id ) const;
    TagList get_labels( const ExpressionID& id ) const;

    /** @brief query if an expression with the given <code>Tag</code> exists in the registry. */
    bool have_entry( const Tag& label ) const;

    /** @brief query if an expression with the given <code>ExpressionID</code> exists in the registry. */
    bool have_entry( const ExpressionID& id ) const;

    /** @brief Print the contents of the registry to the specified output stream. */
    void dump_registry( std::ostream& os ) const;

    /**
     *  @brief Add a new entry to the registry.  Returns true if the
     *  entry was successfully added.
     */
    bool add_entry( const ExpressionID& id,
                    const Tag& label,
                    const bool allowOverWrite=false );

    /**
     *  @brief Remove an entry from the registry.  Returns true if the
     *  entry was successfully removed.
     */
    bool remove_entry( const ExpressionID& id );
    bool remove_entries( const TagList& tags );

    ExpressionRegistry(){}; ///< should only be constructed by an <code>ExpressionFactory</code>
    ~ExpressionRegistry(){};

  private:

    typedef std::multimap<ExpressionID, Tag > IDLabelMap;
    typedef std::map     <Tag, ExpressionID > LabelIDMap;

    IDLabelMap idLabelMap_;
    LabelIDMap labelIDMap_;

    void check_unique_entry( const ExpressionID&, const std::string ) const;
  };

  //--------------------------------------------------------------------
  ExpressionID
  ExpressionRegistry::get_id( const Tag& label ) const
  {
    const LabelIDMap::const_iterator iid = labelIDMap_.find(label);
    if( iid == labelIDMap_.end() ){
      std::ostringstream errmsg;
      errmsg << "ERROR: Could not find an ID for Expression with label: " << std::endl
             << "      '" << label << "'" << std::endl
             << "       You must register this Expression with the ExpressionFactory prior to using it." << std::endl
             << "       Registry contents follow." << std::endl << std::endl;
      dump_registry( errmsg );
      throw std::runtime_error( errmsg.str() );
    }
    return iid->second;
  }
  //--------------------------------------------------------------------
  const Tag&
  ExpressionRegistry::get_label( const ExpressionID& id ) const
  {
    check_unique_entry( id, "get_label()" );

    const IDLabelMap::const_iterator ilbl = idLabelMap_.find( id );
    if( ilbl == idLabelMap_.end() ){
      std::ostringstream errmsg;
      errmsg << "ERROR: Could not find a label for Expression with ID: '" << id << "'" << std::endl
             << "       You must register this Expression with the ExpressionFactory before using it." << std::endl
             << "       Registry contents follow." << std::endl;
      dump_registry( errmsg );
      throw std::runtime_error( errmsg.str() );
    }
    return ilbl->second;
  }
  //--------------------------------------------------------------------
  TagList
  ExpressionRegistry::get_labels( const ExpressionID& id ) const
  {
    TagList tags;
    std::pair<IDLabelMap::const_iterator,IDLabelMap::const_iterator> iters = idLabelMap_.equal_range( id );

    if( iters.first == idLabelMap_.end() ){
      std::ostringstream errmsg;
      errmsg << "ERROR: Could not find a label for Expression with ID: '" << id << "'" << std::endl
             << "       You must register this Expression with the ExpressionFactory before using it." << std::endl
             << "       Registry contents follow." << std::endl;
      dump_registry( errmsg );
      throw std::runtime_error( errmsg.str() );
    }

    for( IDLabelMap::const_iterator i=iters.first; i!=iters.second; ++i ){
      tags.push_back( i->second );
    }
    return tags;
  }
  //--------------------------------------------------------------------
  bool
  ExpressionRegistry::add_entry( const ExpressionID& id,
                                 const Tag& label,
                                 const bool allowOverwrite )
  {
    idLabelMap_.insert( std::make_pair(id,label) );
    std::pair<LabelIDMap::iterator,bool> result2 = labelIDMap_.insert( std::make_pair(label,id) );

    if( !result2.second ){
      if( allowOverwrite ){
        remove_entry( id );
        return( add_entry(id,label,false) );
      }
      else{
        std::ostringstream errmsg;
        errmsg << "ERROR from " << __FILE__ << " : " << __LINE__ << std::endl
               << " Failed to add entry '" << label << "' to registry" << std::endl
               << " This is likely because an expression with this label already exists." << std::endl
               << " Set the 'allowOverwrite' flag in ExpressionRegistry::add_entry() to enable re-registration" << std::endl
               << " A list of registered expressions follows:" << std::endl;
        dump_registry( errmsg );
        throw std::runtime_error( errmsg.str() );
      }
    }
    return result2.second;
  }
  //--------------------------------------------------------------------
  bool
  ExpressionRegistry::have_entry( const Tag& label ) const
  {
    return ( labelIDMap_.find(label)!=labelIDMap_.end() );
  }
  //--------------------------------------------------------------------
  bool
  ExpressionRegistry::have_entry( const ExpressionID& id ) const
  {
    return ( idLabelMap_.find(id) != idLabelMap_.end() );
  }
  //--------------------------------------------------------------------
  void
  ExpressionRegistry::dump_registry( std::ostream& os ) const
  {
    using namespace std;

    os.setf(ios::left);
    os << "_______________________________________________________" << endl
       << setw(4) << left << "ID" << setw(30) << left << "Expression Name" << " State" << endl
       << "-------------------------------------------------------"
       << endl;
    for( IDLabelMap::const_iterator ii=idLabelMap_.begin(); ii!=idLabelMap_.end(); ++ii ){
      os << setw(4) << left << ii->first
          << setw(30) << left << ii->second.name() << " "
          << ii->second.context() << endl;
    }
    os << "_______________________________________________________" << endl
       << endl;
  }
  //--------------------------------------------------------------------
  bool
  ExpressionRegistry::remove_entry( const ExpressionID& id )
  {
    check_unique_entry( id, "remove_entry()" );
    size_t n1=0, n2=0;
    std::pair<IDLabelMap::const_iterator,IDLabelMap::const_iterator> iirange = idLabelMap_.equal_range(id);
    for( IDLabelMap::const_iterator ii=iirange.first; ii!=iirange.second; ++ii ){
      n1 += labelIDMap_.erase( ii->second );
    }
    n2 = idLabelMap_.erase( id );
    return (n1>0 && n2>0);
  }
  //--------------------------------------------------------------------
  bool
  ExpressionRegistry::remove_entries( const TagList& tags )
  {
    size_t n1=0, n2=0;
    for( TagList::const_iterator it=tags.begin(); it!=tags.end(); ++it ){
      LabelIDMap::iterator il = labelIDMap_.find( *it );
      if( il == labelIDMap_.end() ) continue;
      std::pair<IDLabelMap::const_iterator,IDLabelMap::const_iterator> iirange = idLabelMap_.equal_range(il->second);
      for( IDLabelMap::const_iterator ii=iirange.first; ii!=iirange.second; ++ii ){
        n1 += labelIDMap_.erase( ii->second );
      }
      n2 += idLabelMap_.erase( il->second );
    }
    return (n1>0 && n2>0);
  }
  //--------------------------------------------------------------------
  void
  ExpressionRegistry::check_unique_entry( const ExpressionID& id, const std::string method ) const
  {
    if( idLabelMap_.count( id ) > 1 ){
      std::pair<IDLabelMap::const_iterator,IDLabelMap::const_iterator> iters = idLabelMap_.equal_range( id );
      std::ostringstream msg;
      msg << std::endl
          << __FILE__ << " : " << __LINE__ << std::endl
          << "  ExpressionRegistry::" << method << " cannot be used for an ExpressionID that has multiple" << std::endl
          << "  fields evaluated by the expression.  Use ExpressionRegistry::get_labels() instead." << std::endl
          << std::endl
          << "  The following fields are evaluated by the expression with id " << id << ":" << std::endl;
      for( IDLabelMap::const_iterator i=iters.first; i!=iters.second; ++i ){
        msg << "    " << i->second << std::endl;
      }
      throw std::runtime_error( msg.str() );
    }
  }

//--------------------------------------------------------------------

ExpressionFactory::ExpressionFactory( const bool log )
  : outputLog_( log ),
    registry_( new ExpressionRegistry() )
{
  patchIDRequired_ = false;
}

//--------------------------------------------------------------------

ExpressionFactory::~ExpressionFactory()
{
  // wipe out all allocated expressions
  for( PatchExprMap::iterator ip=exprMap_.begin(); ip!=exprMap_.end(); ++ip ){
    IDExprMap& idmap=ip->second;
    for( IDExprMap::iterator ii=idmap.begin(); ii!=idmap.end(); ++ii ){
      if( outputLog_ ){
        std::cout << "factory: deleting " << ii->second->get_tags()[0]
                  << " with id: " << ii->first << " on patch " << ip->first << std::endl;
      }
      delete ii->second;
    }
  }
  for( CallBackMap::iterator icb=callBacks_.begin(); icb!=callBacks_.end(); ++icb ){
    delete icb->second;
  }
  delete registry_;
}

//--------------------------------------------------------------------

void
ExpressionFactory::require_patch_id_specification()
{
  patchIDRequired_ = true;
}

//--------------------------------------------------------------------

ExpressionID
ExpressionFactory::register_expression( const ExpressionBuilder* builder,
                                        const bool allowOverWrite )
{
  assert( builder != NULL );

  if( patchIDRequired_ && !didSetPatchID_ ){
    std::ostringstream msg;
    msg << "Error registering expression that computes\n" << builder->get_tags() << std::endl << std::endl
        << "You must provide a patch ID when registering expressions.  This is because\n\t"
        << "ExpressionFactory::require_patch_id_specification() \nwas called at some point.\n\n"
        << __FILE__ << " : " << __LINE__ << std::endl << std::endl;
    throw std::runtime_error( msg.str() );
  }

  const TagList& names = builder->get_tags();

  if( names.empty() ){
    std::ostringstream msg;
    msg << "ERROR! attempted to register an expression that does not apparently compute any fields!"
        << std::endl << "   " << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( msg.str() );
  }
  {
    bool hasInvalidTag = false;
    std::ostringstream msg;
    msg << "ERROR: expression computes a tag with an invalid context!" << std::endl;
    BOOST_FOREACH( const Tag& tag, names ){
      if( tag.context() == INVALID_CONTEXT ){
        hasInvalidTag = true;
        msg << "  " << tag << std::endl;
      }
    }
    if( hasInvalidTag ) throw std::runtime_error( msg.str() );
  }

  // define the ID for this expression.
  ExpressionID id;

  if( outputLog_ ){
    std::cout << "ExpressionFactory : adding ";
    for( TagList::const_iterator inm=names.begin(); inm!=names.end(); ++inm ){
      std::cout << *inm << ", ";
    }
    std::cout << " ID: " << id << std::endl;
  }

  callBacks_.insert( std::make_pair(id,builder) );

  try{
    if( allowOverWrite ){
      registry_->remove_entries(names);
      BOOST_FOREACH( PatchExprMap::value_type& vt, exprMap_ ){
        IDExprMap& idmap = vt.second;
        const IDExprMap::iterator iexpr = idmap.find(id);
        if( iexpr != idmap.end() ){
          delete iexpr->second;
          idmap.erase(iexpr);
        }
      }
    }
    for( TagList::const_iterator inm=names.begin(); inm!=names.end(); ++inm ){
      registry_->add_entry( id, *inm, allowOverWrite );
    }
  }
  catch( std::exception& err ){
    std::ostringstream msg;
    msg << err.what() << std::endl << std::endl
        << "ERROR from " << __FILE__ << " : " << __LINE__ << std::endl
        << " while trying to register expression that calculates:" << std::endl
        << "  " << names << std::endl
        << " It appears that an expression to calculate one or more" << std::endl
        << " of these quantities has already been registered" << std::endl;
    throw std::runtime_error( msg.str() );
  }

  return id;
}

//--------------------------------------------------------------------

ExpressionID
ExpressionFactory::register_expression( const ExpressionBuilder* builder,
                                        const bool allowOverWrite,
                                        const int fmlID )
{
  didSetPatchID_ = true;
  const ExpressionID id = register_expression( builder, allowOverWrite );
  set_field_manager_list_id( id, fmlID );
  didSetPatchID_ = false;
  return id;
}

//--------------------------------------------------------------------

PollerPtr
ExpressionFactory::get_poller( const Tag& tag )
{
  BOOST_FOREACH( PollerPtr p, pollerList_ ){
    if( tag == p->target_tag() ) return p;
  }
  // if we get here, it wasn't found in the list, so build one.
  PollerPtr p( new Expr::Poller(tag) );
  pollerList_.insert( p );
  return p;
}

//--------------------------------------------------------------------

NonBlockingPollerPtr
ExpressionFactory::get_nonblocking_poller( const Tag& tag )
{
  BOOST_FOREACH( NonBlockingPollerPtr p, nonBlockingPollers_ ){
    if( tag == p->target_tag() ) return p;
  }
  // if we get here, it wasn't found in the list, so build one.
  NonBlockingPollerPtr p( new Expr::NonBlockingPoller(tag) );
  nonBlockingPollers_.insert( p );
  return p;
}

//--------------------------------------------------------------------

void
ExpressionFactory::attach_dependency_to_expression( const Tag& srcTermTag,
                                                    const Tag& targetTag,
                                                    const SourceExprOp op )
{
  if( outputLog_ ){
    std::cout << "Making '" << srcTermTag << "' a dependent of '" << targetTag << "'\n";
  }

  // find the ids for the two expressions
  const ExpressionID srcTermID = registry_->get_id(srcTermTag);
  const ExpressionID targetID  = registry_->get_id(targetTag);

  const TagList& srcNames  = registry_->get_labels( srcTermID );
  const TagList& targNames = registry_->get_labels( targetID );

  const TagList::const_iterator isrc  = std::find( srcNames.begin(),  srcNames.end(), srcTermTag );
  const TagList::const_iterator itarg = std::find( targNames.begin(), targNames.end(), targetTag );

  const int ixsrc  = isrc  - srcNames.begin();
  const int ixtarg = itarg - targNames.begin();

  DepInfo info;
  info.srcID = srcTermID;
  info.op = op;
  info.targFieldIndex = ixtarg;
  info.srcTermFieldIndex = ixsrc;
  idSetMap_[ targetID ].insert( info );
}

//--------------------------------------------------------------------

void
ExpressionFactory::attach_modifier_expression( const Tag& modifierTag,
                                               const Tag& targetTag,
                                               const int patchID,
                                               const bool allowOverWrite )
{
  ModifierMap::iterator imod = modifiers_.find( targetTag );
  if( imod == modifiers_.end() ){
    TagList tags;
    IDTagListMap idmap;
    imod = modifiers_.insert( make_pair(targetTag,idmap) ).first;
  }
  if( outputLog_ ){
    std::cout << "attaching modifier expression " << modifierTag << " to " << targetTag << " on patch " << patchID << std::endl;
  }
  IDTagListMap& idmap = imod->second;
  TagList& tl = idmap[patchID];
  TagList::iterator itl = std::find( tl.begin(), tl.end(), modifierTag );
  if( itl != tl.end() ){  // we found a duplicate
    if( allowOverWrite ){
      *itl = modifierTag;
      return;
    }
    else{
     std::ostringstream msg;
     msg << std::endl << __FILE__ << " : " << __LINE__ << std::endl
         << "ERROR: a duplicate modifier:" << std::endl
         << "  " << modifierTag << std::endl
         << "was added to target:" << std::endl
         << "  " << targetTag << std::endl
         << "but overwriting was forbidden." << std::endl;
     throw std::runtime_error( msg.str() );
    }
  }
  tl.push_back( modifierTag );
}

//--------------------------------------------------------------------

void
ExpressionFactory::cleave_from_parents( const ExpressionID& id )
{
  if( !query_expression(id) ){
    std::ostringstream msg;
    msg << std::endl
        << __FILE__ << " : " << __LINE__ << std::endl << std::endl
        << "  from ExpressionFactory::cleave_from_parents( id )" << std::endl
        << "  ERROR: no expression exists for expression with id " << id
        << std::endl << std::endl;
    throw std::runtime_error( msg.str() );
  }
  cleaveFromParents_.insert( id );
}

//--------------------------------------------------------------------

void
ExpressionFactory::cleave_from_children( const ExpressionID& id )
{
  if( !query_expression(id) ){
    std::ostringstream msg;
    msg << std::endl
        << __FILE__ << " : " << __LINE__ << std::endl << std::endl
        << "  from ExpressionFactory::cleave_from_children( id )" << std::endl
        << "  ERROR: no expression exists for expression with id " << id
        << std::endl << std::endl;
    throw std::runtime_error( msg.str() );
  }
  cleaveFromChildren_.insert( id );
}

//--------------------------------------------------------------------

bool
ExpressionFactory::query_expression( const ExpressionID & id ) const
{
  return (callBacks_.find(id) != callBacks_.end());
}

//--------------------------------------------------------------------

bool
ExpressionFactory::remove_expression( const ExpressionID & id )
{
  bool erased = false;

  // remove the callback entry
  CallBackMap::iterator icb = callBacks_.find( id );
  if( icb != callBacks_.end() ){
    delete icb->second;
    callBacks_.erase(icb);
    erased = true;
  }

  // remove the id-expression map entry
  for( PatchExprMap::iterator ip=exprMap_.begin(); ip!=exprMap_.end(); ++ip ){
    IDExprMap& idmap=ip->second;
    IDExprMap::iterator iimp = idmap.find(id);
    if( iimp!=idmap.end() ){
      delete iimp->second;
      idmap.erase(iimp);
    }
  }

  erased = registry_->remove_entry(id);

  return erased;
}
//--------------------------------------------------------------------

ExpressionBase&
ExpressionFactory::retrieve_expression( const Tag& tag,
                                        const int patchID,
                                        const bool mustExist )
{
  return retrieve_expression( get_id(tag), patchID, mustExist );
}

ExpressionBase&
ExpressionFactory::retrieve_expression( const ExpressionID& id,
                                        const int patchID,
                                        const bool mustExist )
{
  return retrieve_internal(id,patchID,mustExist,false);
}

ExpressionBase&
ExpressionFactory::retrieve_modifier_expression( const Tag& tag,
                                                 const int patchID,
                                                 const bool mustExist )
{
  return retrieve_internal(get_id(tag),patchID,mustExist,true);
}

ExpressionBase&
ExpressionFactory::retrieve_internal( const ExpressionID& id,
                                      const int patchID,
                                      const bool mustExist,
                                      const bool isModifier )
{
  ExpressionBase* expr = NULL;

  if( outputLog_ ){
    std::cout << "retrieving expression for: " << get_labels(id) << " on patch " << patchID << std::endl;
  }

  // do we already have one built?
  PatchExprMap::iterator ipm = exprMap_.find( patchID );
  if( ipm == exprMap_.end() ){
    ipm = exprMap_.insert( std::make_pair( patchID, IDExprMap() ) ).first;
  }

  IDExprMap& idmap = ipm->second;
  const IDExprMap::iterator iexpr = idmap.find(id);
  if( iexpr != idmap.end() ){
    expr = iexpr->second;
  }
  else{
    if( mustExist ){
      std::ostringstream msg;
      msg << std::endl
          << __FILE__ << " : " << __LINE__ << std::endl << std::endl
          << "ERROR: no expression exists for expression with id " << id
          << " and label " << registry_->get_labels(id)[0] << std::endl
          << "       and creation of a new expression was forbidden.\n\n"
          << "Registered expressions:\n";
      dump_expressions( msg );
      throw std::runtime_error( msg.str() );
    }
    // need to build a new one...
    CallBackMap::iterator ifcn = callBacks_.find( id );
    if( ifcn == callBacks_.end() ){
      std::ostringstream errmsg;
      errmsg << "ERROR: No create function for expression with ID:" << id
             << " and label " << registry_->get_labels(id)[0]
             << " has been registered!"
             << std::endl;
      throw std::runtime_error( errmsg.str() );
    }

    if( outputLog_ ){
      std::cout << "  building expression " << get_labels(id) << std::endl;
    }

    // build the function and set the fields that it computes.
    expr = ifcn->second->build();
    const TagList& comptags = ifcn->second->get_tags();

    //_______________________________________________________________
    // now that we have the expression built, add it to the map:
    idmap.insert( std::make_pair( id, expr ) );

    //_______________________________________________________________
    // set the field manager list if relevant
    IDFMLMap::const_iterator ifml = idFMLMap_.find(id);
    if( ifml != idFMLMap_.end() ){
      expr->set_field_manager_list_id( ifml->second );
    }

    //_______________________________________________________________
    // cleave as necessary
    if( cleaveFromParents_.find(id) != cleaveFromParents_.end() )
      expr->cleave_from_parents(true);
    if( cleaveFromChildren_.find(id) != cleaveFromChildren_.end() )
      expr->cleave_from_children(true);

    //_______________________________________________________________
    // Handle modifier expressions. These are special. Also, a modifier
    // cannot have a modifier attached to it. So we only look to attach
    // modifiers to expressions that are not modifiers
    if( !isModifier ){

      // Tags on modifiers are only used to identify the expression, not to
      // associate it with a field.  Therefore, we only set tags on expressions
      // that are not modifiers.
      expr->set_computed_tag( comptags );

      // Attach any relevant modifiers to this expression, looking at each of its
      // computed tags to see if a modifier is associated with any one of them.
      BOOST_FOREACH( const Expr::Tag& comptag, comptags ){
        // Modifier expressions are only allowed on non-modifier expressions;
        // hence this if statement.
        ModifierMap::const_iterator imm = modifiers_.find( comptag );
        if( imm != modifiers_.end() ){

          const IDTagListMap& idtl = imm->second;

          TagList modTags; // all of the modifiers that are relevant to this expression
          {
            // look for a modifier on this patch:
            const IDTagListMap::const_iterator ii = idtl.find(patchID);
            if( ii != idtl.end() )  modTags = ii->second;

            // check to see if there are any that were wanted on all patches:
            const IDTagListMap::const_iterator iiall = idtl.find(ALL_PATCHES);
            if( iiall != idtl.end() ){
              const TagList& tl = iiall->second;
              modTags.insert( modTags.begin(), tl.begin(), tl.end() );
            }
          }
          // if we have a modifier, plug it in
          if( !modTags.empty() ){
            BOOST_FOREACH( const Tag& tag, modTags ){
              if( outputLog_ ){
                std::cout << "  attaching modifier " << tag << " to " << comptag << " for patch " << patchID << std::endl;
              }
              ExpressionBase& modifier = retrieve_internal( get_id(tag), patchID, false, true );
              TagList myNameTag; myNameTag.push_back(imm->first);
              modifier.set_computed_tag( myNameTag );

              if( modifier.field_typeid() != expr->field_typeid() ){
                std::ostringstream msg;
                msg << __FILE__ << " : " << __LINE__ << std::endl
                    << "ERROR: modifier expression: " << modifier.get_tags()[0] << std::endl
                    << "       has a different field type than the target expression: " << comptag << std::endl;
                throw std::runtime_error( msg.str() );
              }

              if( expr->is_gpu_runnable() ){ // Non-Modifier Expression
                if( modifier.is_gpu_runnable() ){ // Modifier Expression
                  if( outputLog_ ){
                    std::cout << "Expression and Modifier expression is tagged GPU runnable. \n";
                  }
                }
                else {
                  expr->set_gpu_runnable(false);
                  if( outputLog_ ){
                    std::cout << "Warning: Modifier Expression " << tag << " attached to" << std::endl
                        << "         Expression: " << comptag << " is not GPU runnable" << std::endl
                        << "         Expression: " << comptag << " has been modified to CPU runnable" << std::endl
                        << "         to comply with Modifier Expression tag." << std::endl;
                  }
                }
              }
              else{
                if( modifier.is_gpu_runnable() ){
                  modifier.set_gpu_runnable(false);
                  if( outputLog_ ){
                    std::cout << "Warning: Modifier Expression " << tag << " is GPU runnable\n"
                        << "         but the expression it is attached to " << comptag
                        << "         is not GPU runnable. The modifier has been changed to CPU runnable\n";
                  }
                }
                else{
                  if( outputLog_ ){
                    std::cout << "Expression and Modifier expression is tagged CPU runnable. \n";
                  }
                }
              }
              expr->add_modifier( &modifier, tag );
            }
          }
        }
      } // BOOST_FOREACH
    } // modifier

    // if any source expressions were attached, inform this expression.
    IDSetMap::const_iterator ids = idSetMap_.find( id );
    if( ids!=idSetMap_.end() ){
      const IDSet& idset = ids->second;
      for( IDSet::const_iterator i=idset.begin(); i!=idset.end(); ++i ){
        ExpressionBase& childExpr = retrieve_expression( i->srcID, patchID );
        if( childExpr.field_typeid() != expr->field_typeid() ){
          std::ostringstream msg;
          msg << __FILE__ << " : " << __LINE__ << std::endl
              << "ERROR: source expression: " << childExpr.get_tags()[0] << std::endl
              << "       has a different field type than the target expression: " << expr->get_tags()[0] << std::endl;
          throw std::runtime_error( msg.str() );
        }

        if( expr->is_gpu_runnable() ){ // Target Expression
          if( childExpr.is_gpu_runnable() ){ // Source Expression
            if( outputLog_ ){
              std::cout << "Source and Target expression is tagged GPU runnable. \n";
            }
          }else {
            expr->set_gpu_runnable(false);
            if( outputLog_ ){
              std::cout << "Warning - Source Expression :" << childExpr.get_tags()[0] << "attached to" << std::endl
                  << "Target Expression :" << expr->get_tags()[0] << "is not GPU runnable" << std::endl
                  << "Target Expression :" << expr->get_tags()[0] << "tag has been modified to CPU runnable" << std::endl
                  << "to comply with Source Expression tag." << std::endl;
            }
          }
        } else{
          if( childExpr.is_gpu_runnable() ){
            childExpr.set_gpu_runnable(false);
            if( outputLog_ ){
              std::cout << "Warning - Source Expression :" << childExpr.get_tags()[0] << "attached to" << std::endl
                  << "Target Expression :" << expr->get_tags()[0] << "is GPU runnable" << std::endl
                  << "Source Expression :" << childExpr.get_tags()[0] << "tag has been modified to CPU runnable" << std::endl
                  << "to comply with Target Expression tag." << std::endl;
            }
          } else{
            if( outputLog_ ){
              std::cout << "Source and Target expression is tagged CPU runnable. \n";
            }
          }
        }

        expr->attach_source_expression( &childExpr, i->op, i->targFieldIndex, i->srcTermFieldIndex );

      } // loop over source expressions
    } // if sources are found

  } // creating a new expression

  return *expr;
}
//--------------------------------------------------------------------

void ExpressionFactory::dump_expressions( std::ostream& os ) const
{
  registry_->dump_registry(os);
}

const Tag& ExpressionFactory::get_label( const ExpressionID& id ) const
{
  return registry_->get_label(id);
}

TagList ExpressionFactory::get_labels( const ExpressionID& id ) const
{
  return registry_->get_labels(id);
}

ExpressionID ExpressionFactory::get_id( const Tag& tag ) const
{
  return registry_->get_id(tag);  // jcs don't we need the patch ID also?
}

bool ExpressionFactory::have_entry( const Tag& tag ) const
{
  return registry_->have_entry(tag);
}

//--------------------------------------------------------------------

std::pair<bool,int>
ExpressionFactory::get_associated_fml_id( const int patchID, const Tag& tag ) const
{
  const IDFMLMap::const_iterator i = idFMLMap_.find( get_id(tag) );
  if( i == idFMLMap_.end() ) return std::make_pair( false, -9999 );
  return std::make_pair( true, i->second );
}

//--------------------------------------------------------------------

void
ExpressionFactory::set_field_manager_list_id( const ExpressionID& id, const int listID )
{
  idFMLMap_[id] = listID;
}

//--------------------------------------------------------------------

} // namespace Expr
