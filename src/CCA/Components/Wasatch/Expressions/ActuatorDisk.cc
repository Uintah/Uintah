#include <CCA/Components/Wasatch/Expressions/ActuatorDisk.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/TagNames.h>


#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>


template< typename FieldT >
ActuatorDisk<FieldT>::
ActuatorDisk( const Expr::Tag& volFracTag,
                const double payload,
                const int rotors,
                const double thrustDir,
                const double radius )
 :  Expr::Expression<FieldT>(), 
    payloadMass_(payload),
    rotors_(rotors), 
    thrustDir_(thrustDir),
    radius_(radius)
 {
     volFrac_ = this->template create_field_request<FieldT>(volFracTag);
 }


 //----------------------------------------

template< typename FieldT >
ActuatorDisk<FieldT>::
~ActuatorDisk()
{}

//-----------------------------------------
template< typename FieldT >
void
ActuatorDisk<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  patchContainer_ = opDB.retrieve_operator<WasatchCore::UintahPatchContainer>();
}

//-----------------------------------------

template< typename FieldT >
void
ActuatorDisk<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  using namespace Uintah;

  const Uintah::Patch* const patch = patchContainer_->get_uintah_patch();
  const double dz =  patch->dCell().z();

  FieldT& result = this->value();
  const double gravityConst = 9.81;
  const double volume = 3.14159 * radius_ * radius_ * 0.025;  //pi*r^2*h where h is dz
  result <<= thrustDir_ * payloadMass_ * gravityConst / rotors_ / volume;
  result <<= volFrac_->field_ref()*result;
}

//-----------------------------------------

template< typename FieldT >
ActuatorDisk<FieldT>::Builder::
Builder(  const Expr::Tag& result,
          const Expr::Tag& volFracTag,
          const double payloadmass,
          const int numrotors,
          const double thrustdir,
          const double rotor_radius)
    : ExpressionBuilder(result),
      volfract_ (volFracTag),
      payloadmass_ (payloadmass),
      numrotors_ (numrotors),
      thrustdir_ (thrustdir),
      rotor_radius_(rotor_radius)
      {}


//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ActuatorDisk<FieldT>::Builder::build() const
{
  return new ActuatorDisk<FieldT>( volfract_, payloadmass_, numrotors_, thrustdir_, rotor_radius_);
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class ActuatorDisk< SpatialOps::SVolField >;
template class ActuatorDisk< SpatialOps::XVolField >;
template class ActuatorDisk< SpatialOps::YVolField >;
template class ActuatorDisk< SpatialOps::ZVolField >;
//==================================================================
