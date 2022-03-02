/**
 * \file OldVariable.cc
 *
 * The MIT License
 *
 * Copyright (c) 2013-2021 The University of Utah
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

#include "OldVariable.h"
#include "FieldAdaptor.h"
#include "Wasatch.h"

#include <boost/foreach.hpp>
#include <string>
#include <vector>

//-- ExprLib includes --//
#include <expression/ExpressionFactory.h>

//-- SpatialOps includes --//

//-- Uintah Includes --//
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace WasatchCore {

  //==================================================================
  /**
   * @brief a helper function used to split a string based on a given delimiter
   * 
   * @param to_split string to split
   * @param delimiter string delimiter
   * @return std::vector<std::string> 
   */
  std::vector<std::string> split(std::string to_split, std::string delimiter) {
      size_t pos = 0;
      std::vector<std::string> matches{};
      do {
          pos = to_split.find(delimiter);
          int change_end;
          if (pos == std::string::npos) {
              pos = to_split.length() - 1;
              change_end = 1;
          }
          else {
              change_end = 0;
          }
          matches.push_back(to_split.substr(0, pos+change_end));
          
          to_split.erase(0, pos+delimiter.size());

      }
      while (!to_split.empty());
      return matches;

  }

  /**
   * @brief This a function that is used in the construction of the Tagname of variable being saved. 
   * 
   * @param old_name string of the variable name
   * @param delimiter string of the delimiter being used to split the old name
   * @return std::string 
   */
  std::string construct_name(std::string old_name,std::string delimiter){
      std::string new_name;
      // this part is used to avoid messing with "_old" suffix in existing code.
      // eventually this has to be changed to "_NM_0" 
      //----------------------------------------------------------------------------------------------------
      std::vector<std::string> splitted_name = split(old_name,"_NM_");
      if (splitted_name.size()==2 ) new_name = splitted_name[0] + "_NM_" + std::to_string(std::stoi(splitted_name.back())+1);
      else if (split(old_name,"_").back()=="old") new_name = split(old_name,"_old").at(0) + "_NM_1";
      else new_name = old_name + "_old";
      return new_name;
  }

  /*
   * \brief Helper function that creates the tag for an old variable
  */
  Expr::Tag create_old_var_tag(const Expr::Tag& varTag, const bool retainVarName, const bool recentValue = false)
  {
    if (recentValue) return Expr::Tag( varTag.name() + ( (retainVarName) ? "" : "_recent" ), Expr::STATE_NONE);
    else{
      if (retainVarName) return Expr::Tag( varTag.name(), Expr::STATE_NONE);
      else return Expr::Tag( construct_name(varTag.name(),"_"), Expr::STATE_NONE);
      } 
  }
  
  /**
   * @brief Create an old variable taglist object for a number (num) of timestep back in time.
   * 
   * @param varTag the tag of the variable of interest
   * @param num number of placeholder for the variable values from previous timestep. 
   * @return Expr::TagList 
   * 
   * example: if we save the pressure from the previous three timesteps, the return taglist from this function will be:
   *  pressure, pressure_NM_0, and pressure_NM_1
   */
  Expr::TagList create_old_var_tag_list(const Expr::Tag& varTag, const int& num)
  {
    Expr::TagList oldVarTags;
    oldVarTags.push_back(varTag);
    for (int i = 1; i<num; i++)
        oldVarTags.push_back(create_old_var_tag(oldVarTags[i-1],false));
    return oldVarTags;
  }

  /**
   * @brief Get the old variable taglist object for a number (num) of timestep back in time
   *        to be used in an expression when creating field requests.
   * 
   * 
   * @param varTag the tag of the variable that was saved for multiple of timesteps
   * @param num the number of timestep for which this variable was saved
   * @return Expr::TagList 
   * 
   * example: if the pressure was saved from the previous three timesteps, the return taglist from this function will be:
   *  pressure_NM_0, pressure_NM_1, and pressure_NM_2
   */
  Expr::TagList get_old_var_taglist(const Expr::Tag& varTag, int num)
  {
    Expr::TagList oldVarTags;
    oldVarTags.push_back(create_old_var_tag(varTag,false));
    for (int i = 1; i<num; i++)
        oldVarTags.push_back(create_old_var_tag(oldVarTags[i-1],false));
    return oldVarTags;
  }

  class OldVariable::VarHelperBase
  {
  protected:
    const Expr::Tag name_, oldName_;
    bool needsNewVarLabel_, recentValue_;
    Uintah::VarLabel *oldVarLabel_, *varLabel_;  // jcs need to get varLabel_ from an existing one...
    Uintah::Ghost::GhostType ghostType_;
    // note that we can pull a VarLabel from the Expr::FieldManager if we have that available.  But we will need a tag and not a string to identify the variable.

  public:

    VarHelperBase( const Expr::Tag& var,
                   const bool retainName,
                   const bool recentValue,
                   const Uintah::TypeDescription* typeDesc,
                   Uintah::Ghost::GhostType ghostType)
    : name_( var ),
      oldName_( create_old_var_tag(name_,retainName, recentValue) ),
      recentValue_(recentValue),
      needsNewVarLabel_ ( Uintah::VarLabel::find( name_.name() ) == nullptr ),
      oldVarLabel_( Uintah::VarLabel::create( oldName_.name(), typeDesc ) ),
      varLabel_   ( needsNewVarLabel_ ? Uintah::VarLabel::create( name_.name(), typeDesc ) : Uintah::VarLabel::find( name_.name() ) ),
      ghostType_  ( ghostType )
    {}

    virtual ~VarHelperBase()
    {
      Uintah::VarLabel::destroy( oldVarLabel_ );
      if (needsNewVarLabel_) Uintah::VarLabel::destroy( varLabel_ );
    }

    const Expr::Tag& var_name()     const{ return name_;    }
    const Expr::Tag& old_var_name() const{ return oldName_; }
    const bool recentValue() const { return recentValue_; }

    Uintah::VarLabel* const  get_var_label    () const{ return varLabel_;    }
    Uintah::VarLabel* const  get_old_var_label() const{ return oldVarLabel_; }
    Uintah::Ghost::GhostType get_ghost_type() { return ghostType_; }

    virtual void populate_old_variable( const AllocInfo& ainfo, const int rkStage ) = 0;
  };

  //==================================================================

  template< typename T >
  class VarHelper : public OldVariable::VarHelperBase
  {
  public:

    VarHelper( const Expr::Tag& var,
               const bool retainName,
               const bool recentValue)
    : VarHelperBase( var,
                     retainName,
                     recentValue,
                     get_uintah_field_type_descriptor<T>(),
                     get_uintah_ghost_type<T>())
    {}

    ~VarHelper(){}

    void populate_old_variable( const AllocInfo& ainfo, const int rkStage )
    {
      using SpatialOps::operator <<=;
      typedef typename SelectUintahFieldType<T>::const_type ConstUintahField;
      typedef typename SelectUintahFieldType<T>::type       UintahField;
      typedef typename SpatialOps::SpatFldPtr<T>            TPtr;

      const Uintah::Ghost::GhostType gt = get_uintah_ghost_type<T>();
      const int ng = 0;  // no ghost cells

      UintahField oldVal; // we will save the current value into this one
      ConstUintahField val; // copy the value from this
      
      if (rkStage == 1) ainfo.newDW->allocateAndPut( oldVal, oldVarLabel_, ainfo.materialIndex, ainfo.patch, gt, ng );
      else              ainfo.newDW->getModifiable ( oldVal, oldVarLabel_, ainfo.materialIndex, ainfo.patch, gt, ng );

      const SpatialOps::GhostData gd( get_n_ghost<T>() );
      TPtr fOldVal = wrap_uintah_field_as_spatialops<T>(oldVal,ainfo,gd);
      
      Uintah::DataWarehouse* dw;
      if (recentValue()) dw = (rkStage == 1) ? ainfo.oldDW : ainfo.newDW;
      else dw = ainfo.oldDW;
      if (dw->exists(varLabel_,ainfo.materialIndex,ainfo.patch)) {
        dw->get( val, varLabel_, ainfo.materialIndex, ainfo.patch, gt, ng );
        const TPtr f = wrap_uintah_field_as_spatialops<T>(val,ainfo,gd);
        (*fOldVal) <<= (*f);
      }
      else  {
        (*fOldVal) <<= 0.0;
      }      
    }
  };

  //==================================================================

  OldVariable::OldVariable()
  {
    wasatchSync_  = false;
    hasDoneSetup_ = false;
    wasatch_ = nullptr;
  }

  //------------------------------------------------------------------

  OldVariable&
  OldVariable::self()
  {
    static OldVariable ov;
    return ov;
  }

  //------------------------------------------------------------------

  OldVariable::~OldVariable()
  {
    BOOST_FOREACH( VarHelperBase* vh, varHelpers_ ){
      delete vh;
    }
  }

  //------------------------------------------------------------------

  template< typename T >
  void
  OldVariable::add_variable( const Category category,
                             const Expr::Tag& var,
                             const bool retainName,
                             const bool recentValue )
  {
    if( hasDoneSetup_ ){
      std::ostringstream msg;
      msg << "OldVariable error: cannot add new variables after tasks have been registered!" << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    if( !wasatchSync_ ){
      std::ostringstream msg;
      msg << "OldVariable error: must call sync_with_wasatch() prior to adding variables!" << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    assert( wasatch_ != nullptr );
    
    // if this expression has already been registered, then return
    Expr::ExpressionFactory& factory = *(wasatch_->graph_categories()[category]->exprFactory);
    const Expr::Tag oldVarTag = create_old_var_tag(var, retainName, recentValue);
    if( factory.have_entry( oldVarTag ) )
    {
      return;
    }

    VarHelperBase* const vh = new VarHelper<T>(var, retainName, recentValue);
    typedef typename Expr::PlaceHolder<T>::Builder PlaceHolder;
    
    const Expr::ExpressionID id =factory.register_expression( new PlaceHolder( vh->old_var_name() ), false );
    wasatch_->graph_categories()[category]->rootIDs.insert(id);
    varHelpers_.push_back( vh );

    // don't allow the ExpressionTree to reclaim memory for this field since
    // it will need to be seen by the task that copies it to the "old" value.
    if ( wasatch_->persistent_fields().find( var.name() ) == wasatch_->persistent_fields().end() )  {
      wasatch_->persistent_fields().insert( var.name() );
    }
  }

  template< typename T >
  void
  OldVariable::add_variables(const Category category,
                             const Expr::Tag& varTag,
                             const int& numVal)
  {
    const Expr::TagList& varTagList = create_old_var_tag_list(varTag,numVal);
    for( Expr::TagList::const_iterator iTag = varTagList.begin(); iTag != varTagList.end(); ++iTag )
    {
      const Expr::Tag tag = * iTag;
      add_variable<T>(category, tag);
    }
  }
  //------------------------------------------------------------------

  void
  OldVariable::sync_with_wasatch( Wasatch* const wasatch )
  {
    wasatch_     = wasatch;
    wasatchSync_ = true;
  }

  //------------------------------------------------------------------

  void
  OldVariable::setup_tasks( const Uintah::PatchSet* const patches,
                            const Uintah::MaterialSet* const materials,
                            Uintah::SchedulerP& sched,
                            const int rkStage )
  {
    if( varHelpers_.size() == 0 ) return;

    // create the Uintah task to accomplish this.
    Uintah::Task* oldVarTask = scinew Uintah::Task( "set old variables", this, &OldVariable::populate_old_variable, rkStage );
    
    BOOST_FOREACH( VarHelperBase* vh, varHelpers_ ){
      oldVarTask->requires( Uintah::Task::OldDW, vh->get_var_label(), vh->get_ghost_type() );
      if (rkStage == 1) oldVarTask->computes( vh->get_old_var_label() );
      else              oldVarTask->modifies( vh->get_old_var_label() );
    }
    sched->addTask( oldVarTask, patches, materials );
    hasDoneSetup_ = true;
  }

  //------------------------------------------------------------------

  void
  OldVariable::populate_old_variable( const Uintah::ProcessorGroup* const pg,
                                      const Uintah::PatchSubset* const patches,
                                      const Uintah::MaterialSubset* const materials,
                                      Uintah::DataWarehouse* const oldDW,
                                      Uintah::DataWarehouse* const newDW,
                                      const int rkStage)
  {
    for( int ip=0; ip<patches->size(); ++ip ){
      const Uintah::Patch* const patch = patches->get(ip);
      for( int im=0; im<materials->size(); ++im ){
        const AllocInfo ainfo( oldDW, newDW, im, patch, pg );
        BOOST_FOREACH( VarHelperBase* vh, varHelpers_ ){
          vh->populate_old_variable( ainfo, rkStage );
        }
      }
    }
  }

  //==================================================================

} /* namespace WasatchCore */



//====================================================================
//-- Explicit template instantiation for supported types

#include "FieldTypes.h"

#define INSTANTIATE( T )                 \
  template class WasatchCore::VarHelper<T>;  \
  template void WasatchCore::OldVariable::add_variable<T>(const Category, const Expr::Tag&, const bool retainName, const bool recentValue); \
  template void WasatchCore::OldVariable::add_variables<T>(const Category, const Expr::Tag&, const int&);

#define INSTANTIATE_VARIANTS( VOL )                            \
  INSTANTIATE( VOL )

INSTANTIATE_VARIANTS( SVolField )
INSTANTIATE_VARIANTS( XVolField )
INSTANTIATE_VARIANTS( YVolField )
INSTANTIATE_VARIANTS( ZVolField )

//INSTANTIATE( ParticleField ) // jcs not ready for particle fields yet.
//====================================================================

