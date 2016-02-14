#ifndef Uintah_Component_Arches_ScalarRHS_h
#define Uintah_Component_Arches_ScalarRHS_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class Operators;
  class Discretization_new;
  class ScalarRHS : public TaskInterface {

public:

    ScalarRHS( std::string task_name, int matl_index );
    ~ScalarRHS();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                      SpatialOps::OperatorDatabase& opr );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                     SpatialOps::OperatorDatabase& opr );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
               SpatialOps::OperatorDatabase& opr );

    void create_local_labels();

    //Build instructions for this (ScalarRHS) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      ScalarRHS* build()
      { return scinew ScalarRHS( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    std::string _rhs_name;
    std::string _D_name;
    std::string _Fconv_name;
    std::string _Fdiff_name;
    Discretization_new* _disc;
    std::string _conv_scheme;

    bool _do_conv;
    bool _do_diff;
    bool _do_clip;

    double _low_clip;
    double _high_clip;


    struct SourceInfo{
      std::string name;
      double weight;
    };
    std::vector<SourceInfo> _source_info;

    WasatchCore::ConvInterpMethods _limiter_type;

    template <class faceT, class velT, class divT>
    void compute_convective_flux( SpatialOps::OperatorDatabase& opr,
                                  const divT* div,
                                  const SpatialOps::SpatFldPtr<velT> u,
                                  const SpatialOps::SpatFldPtr<velT> areaFrac,
                                  SpatialOps::SpatFldPtr<SpatialOps::SVolField> phi,
                                  SpatialOps::SpatFldPtr<SpatialOps::SVolField> rho,
                                  SpatialOps::SpatFldPtr<SpatialOps::SVolField> eps,
                                  SpatialOps::SpatFldPtr<SpatialOps::SVolField> Fconv );

  };

  template <class faceT, class velT, class divT> void
  ScalarRHS::compute_convective_flux(  SpatialOps::OperatorDatabase& opr,
                                       const divT* div,
                                       const SpatialOps::SpatFldPtr<velT> u,
                                       const SpatialOps::SpatFldPtr<velT> areaFrac,
                                       SpatialOps::SpatFldPtr<SpatialOps::SVolField> phi,
                                       SpatialOps::SpatFldPtr<SpatialOps::SVolField> rho,
                                       SpatialOps::SpatFldPtr<SpatialOps::SVolField> eps,
                                       SpatialOps::SpatFldPtr<SpatialOps::SVolField> Fconv )
  {

    //interp the XVol to XSurf
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, velT, faceT>::type InterpVFtoSF;
    const InterpVFtoSF* const interp_xv_to_xf = opr.retrieve_operator<InterpVFtoSF>();
    SpatialOps::SpatFldPtr<faceT> uf = SpatialOps::SpatialFieldStore::get<faceT>( *u );
    SpatialOps::SpatFldPtr<faceT> af_face = SpatialOps::SpatialFieldStore::get<faceT>( *u );

    *uf <<= (*interp_xv_to_xf)(*u);
    *af_face <<= (*interp_xv_to_xf)(*areaFrac);

    //create a volume to face interpolant
    typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SpatialOps::SVolField, faceT>::type InterpSVtoSF;
    const InterpSVtoSF* const i_v_to_s = opr.retrieve_operator<InterpSVtoSF>();

    //create some temp variables modeled on ufx
    SpatialOps::SpatFldPtr<faceT> phiLow = SpatialOps::SpatialFieldStore::get<faceT>( *uf );
    SpatialOps::SpatFldPtr<faceT> phiHi = SpatialOps::SpatialFieldStore::get<faceT>( *uf );
    SpatialOps::SpatFldPtr<faceT> psi = SpatialOps::SpatialFieldStore::get<faceT>( *uf );

    //get upwind (low) interpolant and the flux limiter operator
    typedef UpwindInterpolant<SpatialOps::SVolField, faceT> Upwind;
    typedef FluxLimiterInterpolant<SpatialOps::SVolField, faceT> FluxLim;
    Upwind* upwind = opr.retrieve_operator<Upwind>();
    FluxLim* fluxlim = opr.retrieve_operator<FluxLim>();

    //compute the central differenced term
    //*phiHi <<= (*i_v_to_s)(*phi);
    *phiHi <<= (*i_v_to_s)(*rho * *phi);

    //compute the upwind differenced term
    upwind->set_advective_velocity( *uf );
    upwind->apply_to_field( *phi, *phiLow );
    *phiLow <<= (*i_v_to_s)(*rho) * *phiLow ;

    //compute the flux limiter psi function
    fluxlim->set_advective_velocity( *uf );
    fluxlim->set_flux_limiter_type( _limiter_type );
    fluxlim->apply_to_field( *phi, *psi );
    fluxlim->apply_embedded_boundaries( *eps, *psi );

    //compute the convective term and sum it in
    //*Fconv <<= *Fconv + (*div)( *uf * *af_face * (*i_v_to_s)(*rho) * ( *phiLow - *psi * (*phiLow - *phiHi)));
    *Fconv <<= *Fconv + (*div)( *uf * *af_face * ( *phiLow - *psi * (*phiLow - *phiHi)));

  }
}
#endif
