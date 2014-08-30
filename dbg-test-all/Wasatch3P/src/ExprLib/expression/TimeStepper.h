/**
 *  \file   TimeStepper.h
 *  \author James C. Sutherland
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

#ifndef Expr_TimeStepper_h
#define Expr_TimeStepper_h

#include <set>
#include <fstream>

#include <spatialops/SpatialOpsConfigure.h> // defines thread stuff
#include <spatialops/Nebo.h>

#include <expression/ExpressionTree.h>
#include <expression/FieldManagerList.h>
#include <expression/ExpressionFactory.h>
#include <expression/FieldManager.h>
#include <expression/FieldManagerList.h>

#ifdef ENABLE_THREADS
#  include <spatialops/ThreadPool.h>
#  include <boost/bind.hpp>
#endif

#include <boost/foreach.hpp>

namespace Expr{

  /**
   *  @enum TSMethod
   *  @brief enumerates the supported time steppers.
   */
  enum TSMethod {
    ForwardEuler, ///< Implements the first order forward euler time integrator.
    SSPRK3        ///< Implements the SSP RK3 third order RK time integrator.
  };

  // forward declarations:
  namespace TSLocal{
    class VariableStepperBase;
    template<typename FieldT> class FEVarStepper;
    template<typename FieldT> class SSPRK3VarStepper;
  }


  /**
   *  @class  TimeStepper
   *  @author James C. Sutherland
   *  @date   March, 2008
   *  @brief  Explicit time stepping compatible with expressions.
   *
   *  The TimeStepper class provides a way to perform explicit time
   *  integration using the Expression approach.  It works with
   *  heterogeneous Expression types.  In other words, it can integrate
   *  fields that are of different types simultaneously.  The
   *  TimeStepper class manages field registration and also builds the
   *  appropriate ExpressionTree to evaluate the RHS for all fields that
   *  it integrates.
   *
   *  \par How to use the TimeStepper
   *   \li Register all Expression and SpatialOperator objects that are needed.
   *   \li Create a Patch.
   *   \li Build a TimeStepper
   *   \li Add all desired equations via the
   *       <code>TimeStepper::add_equation()</code> method.
   *   \li Call <code>TimeStepper::finalize()</code> to register all fields
   *       and bind them to all of the relevant expressions.
   *   \li Integrate forward in time.
   *
   *  Example:
   *   \code
   *   TimeStepper integrator( patch );
   *   integrator.add_equation( "solnVar1", rhsExprID1 );
   *   integrator.add_equation( "solnVar2", rhsExprID2 );
   *   integrator.finalize();
   *   while( time < time_end ){
   *      integrator.step( timestep );
   *      time += timestep;
   *   }
   *   \endcode
   *
   *  \par The Simulation Time
   *   The TimeStepper class registers a variable "time" with state
   *   Expr::STATE_NONE that contains the solution time.
   *
   *  @todo Need to set the simulation time at each stage of the
   *        integrator - not only on each step.
   *
   *  @todo - Consider templating this class on the stepper type.  Then
   *  we could implement various RK flavors (low-storage).
   *
   *  @todo Need to deal with tree splitting.  Currently this is not
   *  supported in the integrator.
   */
  class TimeStepper
  {
    typedef SpatialOps::SingleValueField DoubleVal;

  public:

    /**
     *  @brief Create a TimeStepper
     *  @param factory The ExpressionFactory where the expressions are registered.
     *  @param method The time integration method (see TSMethod enum)
     *  @param patchID the ID for the patch that this TimeStepper lives on.
     *  @param timeTag the tag describing the time variable.  Defaults to "time"
     */
    inline TimeStepper( ExpressionFactory& factory,
                        const TSMethod method=ForwardEuler,
                        const int patchID=0,
                        const Tag timeTag = Tag( "time", STATE_NONE ) );

    inline virtual ~TimeStepper();

    inline int patch_id() const{ return masterTree_->patch_id(); }

    /** @brief Get the ExpressionTree that the timestepper is using */
    inline const ExpressionTree* get_tree() const{ return masterTree_; }
    inline ExpressionTree* get_tree()      { return masterTree_; }

    /** @brief Get the list of variable names that are being solved on
     *  this TimeStepper
     */
    inline void get_variables( std::vector<std::string>& variables ) const;

    /** @brief set the time */
    inline void set_time( const double time )
    {
      using namespace SpatialOps;
      assert(isFinalized_);
      BOOST_FOREACH( DoubleVal* t, simtime_ ){
        *t <<= time;
      }
      timeShadow_ = time;
    }

    /**
     * @return the current time
     */
    inline double get_time() const
    {
      return timeShadow_;
    }

    /**
     * @brief Obtain pairs of solution variables and their typeid names (in that order)
     */
    inline void get_name_type_pairs( std::vector<std::pair<std::string,std::string> >& ) const;

    /** @brief Get the vector of Expr::Tag objects for the variables
     *  solved on this TimeStepper
     */
    inline void get_rhs_tags( std::vector<Tag>& rhsTags ) const;

    /** @brief Take a single timestep */
    inline void step( const double timeStep );

    /**
     *  @brief Add an equation to the timestepper.
     *
     *  @param solnVarName The name of the variable that is to be
     *  integrated in time.  This variable will be automatically
     *  registered.
     *
     *  @param rhsTag The Tag for the Expression that calculates the
     *  right-hand-side function for this equation.
     *
     *  @param fmlID for use in situations where multiple FieldManagerLists are
     *  employed, this identifies which one should be associated with this RHS.
     */
    template<typename FieldT>
    void add_equation( const std::string solnVarName,
                       const Tag rhsTag,
                       const int fmlID=-99999 )
    {
      // build the VariableStepper for this equation and stash it.
      switch (method_){
      case ForwardEuler: varSet_.insert( new TSLocal::FEVarStepper    <FieldT>( factory_, fmlID, solnVarName, rhsTag, timeTag_ ) ); break;
      case SSPRK3:       varSet_.insert( new TSLocal::SSPRK3VarStepper<FieldT>( factory_, fmlID, solnVarName, rhsTag, timeTag_ ) ); break;
      }
      const ExpressionID rhsID = factory_.get_id(rhsTag);
      exprIDs_.insert( rhsID );      // save off the ID for this equation
    }

    /**
     *  @brief Add an equation to the timestepper.
     *
     *  @param solnVarName The name of the variable that is to be
     *  integrated in time.  This variable will be automatically
     *  registered.
     *
     *  @param rhsID The ExpressionID for the Expression that calculates
     *  the right-hand-side function for this equation.
     *
     *  @param fmlID for use in situations where multiple FieldManagerLists are
     *  employed, this identifies which one should be associated with this RHS.
     */
    template<typename FieldT>
    void add_equation( const std::string solnVarName,
                       const ExpressionID rhsID,
                       const int fmlID=-99999 )
    {
      this->add_equation<FieldT>( solnVarName, factory_.get_label(rhsID), fmlID );
    }

    /**
     *  @brief Add a group of equations whose RHS is evaluated by a
     *         single expression to the timestepper.
     *
     *  @param solnVarNames a vector of names of the solution
     *         variables.  The order must be exactly the same as the
     *         corresponding RHS evaluation order in the expression
     *         that calculates the RHS for these variables.
     *
     *  @param rhsID The ExpressionID for the Expression that
     *         calculates the RHS for all of these solution variables.
     *
     *  @param fmlID for use in situations where multiple FieldManagerLists are
     *         employed, this identifies which one should be associated with
     *         this RHS.
     */
    template<typename FieldT>
    void add_equations( const std::vector<std::string>& solnVarNames,
                        const ExpressionID rhsID,
                        const int fmlID=-99999 )
    {
      const TagList& tags = factory_.get_labels( rhsID );

      if( tags.size() != solnVarNames.size() ){
        std::ostringstream msg;
        msg << "ERROR: TimeStepper::add_equations() was given a set of solution variable names" << std::endl
            << "       that is inconsistent with the number of RHS evaluated by the RHS Expression ID provided." << std::endl
            << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( msg.str() );
      }

      // build the VariableStepper for this equation and stash it.
      typedef std::vector<std::string> StringVec;
      TagList::const_iterator itg=tags.begin();
      for( StringVec::const_iterator inm=solnVarNames.begin(); inm!=solnVarNames.end(); ++inm, ++itg ){
        switch (method_){
        case ForwardEuler: varSet_.insert( new TSLocal::FEVarStepper    <FieldT>( factory_, fmlID, *inm, *itg, timeTag_ ) ); break;
        case SSPRK3:       varSet_.insert( new TSLocal::SSPRK3VarStepper<FieldT>( factory_, fmlID, *inm, *itg, timeTag_ ) ); break;
        }
      }
      exprIDs_.insert( rhsID );      // save off the ID for this equation
    }
    
    /**
     * @brief Adds a set of ids to the exprIDs set as root nodes of the tree. 
     * 
     * @param ids - the set of expersion ids to be added. 
     * 
     */
    inline void add_ids(ExpressionTree::IDSet ids)
    {
    	assert(!isFinalized_); //can only add up until finalized;
    	exprIDs_.insert(ids.begin(), ids.end());
    }
    /**
     *  @brief After adding all equations to the TimeStepper, this
     *  method should be called.  This results in the variables being
     *  registered, the ExpressionTree being set up, etc.
     *
     *  @param fml - the FieldManagerList that should be associated
     *  with this TimeStepper.
     *
     *  @param opDB - the OperatorDatabase to use in retrieving
     *  operators for the Expression objects on this TimeStepper.
     *
     *  @param t - must play nicely with the FieldManagerList::allocate_fields()
     *   method, which in turn calls through to FieldManagers, passing this object.
     */
    template< typename T >
    inline void finalize( FieldManagerList& fml,
                          const SpatialOps::OperatorDatabase& opDB,
                          const T& t );
    /**
     *  @brief Use this interface when multiple FieldManagerList objects are
     *   being maintained on this TimeStepper.
     *
     *  @param fmls - the FMLMap that should be associated
     *  with this TimeStepper.
     *
     *  @param opDBs - the OperatorDatabase objects to use in retrieving
     *  operators for the Expression objects on this TimeStepper.
     *
     *  @param tmap - Map of FML ids to FieldInfo objects that will be used to
     *   allocate fields on each corresponding FieldManagerList.
     */
    template< typename T >
    inline void finalize( FMLMap& fmls,
                          const OpDBMap& opDBs,
                          std::map<int,const T*>& tmap );

  private:

    const Tag timeTag_;

    typedef std::set<TSLocal::VariableStepperBase*>  VarSet;
    typedef std::vector<ExpressionID> ExprIDVec;
    typedef std::map<ExpressionID,int> ExprFMLIDMap;

    ExpressionFactory&     factory_;
    const TSMethod         method_;       ///< Identifies the method used for time integration
    FieldDeps              fieldDeps_;    ///< A composite list of all fields required for this TimeStepper.
    VarSet                 varSet_;       ///< The set of variables solved on this TimeStepper
    ExpressionTree::IDSet  exprIDs_;      ///< The ids for the rhs expressions corresponding to the variables solved on this TimeStepper.
    ExpressionTree* const  masterTree_;   ///< The tree that is executed to calculate all RHS expressions.
    bool                   isFinalized_;  ///< flag indicating if the integrator has been finalized
    std::vector<DoubleVal*> simtime_;     ///< The time
    double                  timeShadow_;  ///< The time - always on CPU, and tracks simtime_

    TimeStepper(); // no default constructor
    TimeStepper( const TimeStepper& );  // no copying
    TimeStepper& operator=( const TimeStepper& ); // no assignment

  }; // class TimeStepper





  namespace TSLocal{

    class VariableStepperBase
    {
    public:
      VariableStepperBase( const std::string& varName,
                           const Tag& rhsTag,
                           const std::type_info& typeinfo,
                           const Tag& timeTag )
      : typeinfo_  ( typeinfo.name()     ),
        rhsTag_    ( rhsTag              ),
        timeTag_   ( timeTag ),
        varTagNONE_( varName, STATE_NONE ),
        varTagN_   ( varName, STATE_N    ),
        varTagNP1_ ( varName, STATE_NP1  )
      {
        istage_=0;
      }
      virtual ~VariableStepperBase(){}
      virtual void step( const double timestep ) = 0;
      virtual void bind_fields( FieldManagerList& fml ) = 0;
      virtual void set_stage( const int istage, const double dt ){ istage_=istage; }
      const std::string& name() const{ return varTagN_.name(); }
      const Tag& get_name_tag_n() const{ return varTagN_; }
      const Tag& get_name_tag_np1() const{ return varTagNP1_; }
      const Tag& get_name_tag_none() const{ return varTagNONE_; }
      const Tag& rhs_label() const{ return rhsTag_; }
      const std::string type_name() const{ return typeinfo_; }
    protected:
      const std::string typeinfo_;
      const Tag rhsTag_, timeTag_, varTagNONE_, varTagN_, varTagNP1_;
      int istage_;
    private:
      VariableStepperBase(); // no default constructor
      VariableStepperBase& operator=(const VariableStepperBase&); // no assignment
      VariableStepperBase(const VariableStepperBase&);  // no copying
    };


    class ThreadHelper
    {
      TSLocal::VariableStepperBase& vs_;
      const double ts_;
    public:
      ThreadHelper( TSLocal::VariableStepperBase& vs,
                    const double timeStep )
        : vs_(vs), ts_(timeStep) {}
      inline void operator()() const
      { vs_.step(ts_); }
    };


    template< typename FieldT >
    class FEVarStepper : public VariableStepperBase
    {
    public:
      FEVarStepper( ExpressionFactory& factory,
                    const int fmlID,
                    const std::string& solnVarName,
                    const Tag& rhsTag,
                    const Tag& timeTag );

      ~FEVarStepper();

      void step( const double timestep );

      void bind_fields( FieldManagerList& fml );

    private:
      FieldT *varOld_, *varNew_;
      const FieldT *rhs_;
    };


    template< typename FieldT >
    class SSPRK3VarStepper : public VariableStepperBase
    {
    public:
      SSPRK3VarStepper( ExpressionFactory& factory,
                        const int fmlID,
                        const std::string& solnVarName,
                        const Tag& rhsTag,
                        const Tag& timeTag );
      ~SSPRK3VarStepper();
      void step( const double tstep );
      void bind_fields( FieldManagerList& fml );
      void set_stage( const int istage, const double dt );
    private:
      FieldT *varN_, *varOld_, *varNew_;
      const FieldT *rhs_;
      typedef SpatialOps::SingleValueField TimeField;
      TimeField* simtime_;
      SpatialOps::SpatFldPtr<TimeField> time0_;
    };


  } // namespace TSLocal




  //####################################################################
  //
  //
  //                        IMPLEMENTATIONS
  //
  //
  //####################################################################



  //--------------------------------------------------------------------

  template< typename T >
  void
  TimeStepper::finalize( FieldManagerList& fml,
                         const SpatialOps::OperatorDatabase& opDB,
                         const T& t )
  {
    FMLMap fmls; fmls[0]=&fml;
    std::map<int,const T*> tmap;  tmap[0]=&t;
    OpDBMap opDBs; opDBs[0] = &opDB;
    finalize<T>(fmls,opDBs,tmap);
  }

  //--------------------------------------------------------------------

  template< typename T >
  void
  TimeStepper::finalize( FMLMap& fmls,
                         const OpDBMap& opDBMap,
                         std::map<int,const T*>& tmap )
  {
    // ensure that the NP1 and N variable expressions are in the graph.
    // in some cases, they may not be if the RHS does not depend directly on them.
    // This will ensure that the GPU implementations can reason properly with these fields.
    BOOST_FOREACH( TSLocal::VariableStepperBase* vsb, varSet_ ){
      exprIDs_.insert( factory_.get_id( vsb->get_name_tag_n()   ) );
      exprIDs_.insert( factory_.get_id( vsb->get_name_tag_np1() ) );
      if( factory_.have_entry( vsb->get_name_tag_none() ) )
        exprIDs_.insert( factory_.get_id( vsb->get_name_tag_none() ) );
    }

    masterTree_->insert_tree( exprIDs_ );

    //--- register fields
    masterTree_->register_fields( fmls );
    fieldDeps_.requires_field<DoubleVal>( timeTag_ );

    FieldDeps::FldHelpers& rhsfields = fieldDeps_.field_helpers();

    BOOST_FOREACH( FieldDeps::FieldHelperBase* fh, rhsfields ){
      BOOST_FOREACH( FMLMap::value_type fmlpair, fmls ){
        FieldManagerList* const fml = fmlpair.second;
        fh->register_field( *fml );
      }
    }

    //--- allocate fields
    assert( fmls.size() == tmap.size() );
    BOOST_FOREACH( FMLMap::value_type fmlval, fmls ){
      const int id = fmlval.first;
      FieldManagerList* const fml = fmlval.second;
      // extract the appropriate FieldInfo to allocate fields on this fml
      typename std::map<int,const T*>::iterator itm = tmap.find(id);
      assert( itm != tmap.end() );
      fml->allocate_fields(*itm->second);
    }

    //--- bind all fields on the tree and the VariableStepperBase objects
    try{
      masterTree_->bind_fields( fmls );
      BOOST_FOREACH( TSLocal::VariableStepperBase* vsb, varSet_ ){
        const std::pair<bool,int> idpair = factory_.get_associated_fml_id( masterTree_->patch_id(), vsb->rhs_label() );
        FieldManagerList* const fml = extract_field_manager_list( fmls, idpair.second );
        vsb->bind_fields( *fml );
      }
    }
    catch( std::runtime_error& err ){
      std::ofstream out("tree_bind_failed.dot");
      masterTree_->write_tree( out );
      std::ostringstream msg;
      msg << __FILE__ << ":" << __LINE__ << std::endl
          << "Error binding fields."
          << " The expression tree has been written to 'tree_bind_failed.dot'"
          << std::endl << std::endl
          << err.what();
      throw std::runtime_error( msg.str() );
    }

    //--- bind operators to all expressions in the tree
    masterTree_->bind_operators( opDBMap );

    // resolve the time field
    simtime_.clear();
    BOOST_FOREACH( FMLMap::value_type& pr, fmls ){
      using namespace SpatialOps;
      FieldManagerList* fml = pr.second;
      DoubleVal& t = fml->field_ref<DoubleVal>(timeTag_);
      t <<= timeShadow_;
      simtime_.push_back( &t );
    }

    isFinalized_ = true;
  }

  //--------------------------------------------------------------------

  TimeStepper::TimeStepper( ExpressionFactory& factory,
                            const TSMethod method,
                            const int patchID, Tag timeTag )
    : timeTag_( timeTag ),
      factory_( factory ),
      method_( method ),
      masterTree_( new ExpressionTree( factory, patchID ) )
  {
    isFinalized_ = false;
  }

  //--------------------------------------------------------------------

  TimeStepper::~TimeStepper()
  {
    for( VarSet::iterator ivar=varSet_.begin();
         ivar!=varSet_.end();
         ++ivar )
      {
        delete *ivar;
      }
    delete masterTree_;
  }

  //--------------------------------------------------------------------

  void
  TimeStepper::
  get_variables( std::vector<std::string>& variables ) const
  {
    for( VarSet::const_iterator ivar=varSet_.begin(); ivar!=varSet_.end(); ++ivar ){
      variables.push_back( (*ivar)->name() );
    }
  }
  //--------------------------------------------------------------------

  void
  TimeStepper::
  get_name_type_pairs( std::vector<std::pair<std::string,std::string> >& ntp ) const
  {
    for( VarSet::const_iterator ivar=varSet_.begin(); ivar!=varSet_.end(); ++ivar ){
      ntp.push_back( make_pair((*ivar)->name(),(*ivar)->type_name()) );
    }
  }

  //--------------------------------------------------------------------

  void
  TimeStepper::
  get_rhs_tags( std::vector<Tag>& rhsTags ) const
  {
    for( VarSet::const_iterator ivar=varSet_.begin(); ivar!=varSet_.end(); ++ivar ){
      rhsTags.push_back( (*ivar)->rhs_label() );
    }
  }

  //--------------------------------------------------------------------

  void
  TimeStepper::step( const double tstep )
  {
    assert( isFinalized_ ); // jcs should throw.

#   ifdef ENABLE_THREADS
    SpatialOps::ThreadPool& tp = SpatialOps::ThreadPool::self();
#   endif
    switch ( method_ ){
      case ForwardEuler: {
        masterTree_->execute_tree();
        for( VarSet::iterator ivar=varSet_.begin(); ivar!=varSet_.end(); ++ivar ){
#         ifdef ENABLE_THREADS
          const TSLocal::ThreadHelper th(**ivar,tstep);
          tp.schedule( boost::threadpool::prio_task_func(1,th) );
#         else
          (*ivar)->step( tstep );
#         endif
        }
#       ifdef ENABLE_THREADS
        tp.wait();
#       endif

      break;
    }
    case SSPRK3: {
      for( int istage=1; istage<=3; ++istage ){
        for( VarSet::iterator ivar=varSet_.begin(); ivar!=varSet_.end(); ++ivar ){
          (*ivar)->set_stage(istage,tstep);  // note that this sets the simulation time at the appropriate stage.
        }
        masterTree_->execute_tree();
        for( VarSet::iterator ivar=varSet_.begin(); ivar!=varSet_.end(); ++ivar ){
#         ifdef ENABLE_THREADS
          const TSLocal::ThreadHelper th(**ivar,tstep);
          tp.schedule( boost::threadpool::prio_task_func(1,th) );
#         else
          (*ivar)->step( tstep );
#         endif
        }
#       ifdef ENABLE_THREADS
        tp.wait();
#       endif
      }
      break;
    }
    default: assert(0); break;
    } // switch

    using namespace SpatialOps;
    BOOST_FOREACH( DoubleVal* t, simtime_ ){
      *t <<= *t + tstep;
    }
    timeShadow_ += tstep;
  }

  //--------------------------------------------------------------------

  namespace TSLocal{

    template<typename FieldT>
    FEVarStepper<FieldT>::
    FEVarStepper( ExpressionFactory& factory,
                  const int fmlID,
                  const std::string& varName,
                  const Tag& rhsTag,
                  const Tag& timeTag )
      : VariableStepperBase( varName, rhsTag, typeid(FieldT), timeTag )
    {
      typedef typename PlaceHolder<FieldT>::Builder  FieldExpr;
      factory.register_expression( new FieldExpr(varTagN_   ), false, fmlID );
      factory.register_expression( new FieldExpr(varTagNP1_ ), false, fmlID );

      varOld_ = NULL;
      varNew_ = NULL;
      rhs_    = NULL;
    }

    //--------------------------------------------------------------------

    template<typename FieldT>
    FEVarStepper<FieldT>::
    ~FEVarStepper()
    {}

    //--------------------------------------------------------------------

    template< typename FieldT >
    void
    FEVarStepper<FieldT>::
    bind_fields( FieldManagerList& fml )
    {
      typename FieldMgrSelector<FieldT>::type& fmgr = fml.field_manager<FieldT>();
      rhs_    = &fmgr.field_ref( rhsTag_   );
      varOld_ = &fmgr.field_ref( varTagN_  );
      varNew_ = &fmgr.field_ref( varTagNP1_);
    }

    //--------------------------------------------------------------------

    template<typename FieldT>
    void
    FEVarStepper<FieldT>::
    step( const double timestep )
    {
      using namespace SpatialOps;
      // advance variable in time.
      *varNew_ <<= *rhs_ * timestep + *varOld_;
      *varOld_ <<= *varNew_;
    }

    //--------------------------------------------------------------------

    template<typename FieldT>
    SSPRK3VarStepper<FieldT>::
    SSPRK3VarStepper( ExpressionFactory& factory,
                      const int fmlID,
                      const std::string& varName,
                      const Tag& rhsTag,
                      const Tag& timeTag )
      : VariableStepperBase( varName, rhsTag, typeid(FieldT), timeTag )
    {
      typedef typename PlaceHolder<FieldT>::Builder  FieldExpr;
      factory.register_expression( new FieldExpr(Tag(varName,STATE_N   )), false, fmlID );
      factory.register_expression( new FieldExpr(Tag(varName,STATE_NP1 )), false, fmlID );
      factory.register_expression( new FieldExpr(Tag(varName,STATE_NONE)), false, fmlID );

      varN_   = NULL;
      varOld_ = NULL;
      varNew_ = NULL;
      rhs_    = NULL;
    }

    //--------------------------------------------------------------------

    template<typename FieldT>
    SSPRK3VarStepper<FieldT>::
    ~SSPRK3VarStepper()
    {}

    //--------------------------------------------------------------------

    template<typename FieldT>
    void
    SSPRK3VarStepper<FieldT>::
    bind_fields( FieldManagerList& fml )
    {
      typename FieldMgrSelector<FieldT>::type& fmgr = fml.field_manager<FieldT>();
      varN_    = &fmgr.field_ref( varTagNONE_ );
      varOld_  = &fmgr.field_ref( varTagN_    );
      varNew_  = &fmgr.field_ref( varTagNP1_  );
      rhs_     = &fmgr.field_ref( rhsTag_     );

      simtime_ = &fml.field_ref<SpatialOps::SingleValueField>( timeTag_ );
#     ifdef ENABLE_CUDA
      simtime_->set_device_as_active( rhs_->active_device_index() );
#     endif
      time0_   = SpatialOps::SpatialFieldStore::get<TimeField>(*simtime_);
    }

    //--------------------------------------------------------------------

    template<typename FieldT>
    void SSPRK3VarStepper<FieldT>::set_stage( const int istage, const double dt )
    {
      using namespace SpatialOps; // for <<=
      istage_=istage;

      switch( istage ){
        case 1: *time0_   <<= *simtime_;          break;
        case 2: *simtime_ <<= *time0_ + dt;       break;
        case 3: *simtime_ <<= *time0_ + 0.5*dt;   break;
        default: break;
      }
    }

    template<typename FieldT>
    void
    SSPRK3VarStepper<FieldT>::
    step( const double timestep )
    {
      using namespace SpatialOps;

      FieldT& ui   = *varOld_;   // carry forward to next stage (u_{k-1})
      FieldT& un   = *varN_;     // keep as un
      FieldT& unp1 = *varNew_;   // unp1

      switch( istage_ ) {
      case 1:{
        un <<= ui;
        ui <<= ui + *rhs_ * timestep;
        break;
      }
      case 2:{
        ui <<= 0.25*ui + 0.75*un + 0.25*timestep * *rhs_;
        break;
      }
      case 3:{
        ui <<= 2.0/3.0*ui + 1.0/3.0*un + 2.0/3.0*timestep * *rhs_;
        unp1 <<= ui;
        *simtime_ <<= *time0_;  // reset the simulation time since it is advanced elsewhere.
        break;
      }
      default: assert(0);  break;
      } // switch

    }

    //--------------------------------------------------------------------

  } // namespace TSLocal
} // namespace Expr

#endif // Expr_TimeStepper_h
