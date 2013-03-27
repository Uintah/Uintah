#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/WestbrookDryer.h>
#include <Core/Exceptions/ParameterNotFound.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

WestbrookDryer::WestbrookDryer( std::string src_name, ArchesLabel* field_labels,
                            vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, field_labels->d_sharedState, req_label_names, type ), 
  _field_labels(field_labels)
{ 

  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 

  _extra_local_labels.resize(2); 
  std::string tag = "WDstrip_" + src_name; 
  d_WDstrippingLabel = VarLabel::create( tag,  CCVariable<double>::getTypeDescription() ); 
  _extra_local_labels[0] = d_WDstrippingLabel; 
  
  tag = "WDextent_" + src_name; 
  d_WDextentLabel    = VarLabel::create( tag, CCVariable<double>::getTypeDescription() ); 
  _extra_local_labels[1] = d_WDextentLabel; 

  _source_grid_type = CC_SRC; 
  _use_T_clip = false;
  _use_flam_limits = false; 
}

WestbrookDryer::~WestbrookDryer()
{

  VarLabel::destroy( d_WDstrippingLabel ); 
  VarLabel::destroy( d_WDextentLabel ); 

}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
WestbrookDryer::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  // Defaults to methane.
  db->getWithDefault("A",d_A, 1.3e8);     // Pre-exponential factor 
  db->getWithDefault("E_R",d_ER, 24.4e3); // Activation Temperature 
  db->getWithDefault("X", d_X, 1);        // C_xH_y
  db->getWithDefault("Y", d_Y, 4);        // C_xH_y
  db->getWithDefault("m", d_m, -0.3);     // [C_xH_y]^m 
  db->getWithDefault("n", d_n, 1.3 );     // [O_2]^n 
  db->require("fuel_mass_fraction", d_MF_HC_f1);           // Mass fraction of C_xH_y when f=1
  db->require("oxidizer_O2_mass_fraction", d_MF_O2_f0);    // Mass fraction of O2 when f=0
  // labels: 
  db->getWithDefault("temperature_label", d_T_label, "temperature");          // The name of the mixture fraction label
  db->getWithDefault("density_label", d_rho_label, "density");                // The name of the density label 
  db->require("cstar_fraction_label", d_cstar_label);                         // The name of the C* mixture fraction label
  db->require("c_fraction_label", d_ceq_label);                               // The name of the secondary mixture fraciton label
  db->getWithDefault("o2_label", d_o2_label, "O2");                           // The name of the O2 label

  if ( db->findBlock("temperature_clip") ){ 
    db->getWithDefault("temperature_clip", d_T_clip, 10000);                 // [K], Turns off the rate below this T.
    _use_T_clip = true;
  }

  if ( db->findBlock("flammability_limit") ){ 

    _const_diluent = false; 

    if ( db->findBlock("flammability_limit")->findBlock("const_diluent") ){ 

      db->findBlock("flammability_limit")->require("const_diluent",_const_diluent_mass_fraction); 
      _const_diluent = true; 

    } else { 

      db->findBlock("flammability_limit")->findBlock("diluent")->getAttribute("label",_diluent_label_name); 

    }

    db->findBlock("flammability_limit")->findBlock("lower")->getAttribute("slope", _flam_low_m);
    db->findBlock("flammability_limit")->findBlock("lower")->getAttribute("intercept",_flam_low_b);
    db->findBlock("flammability_limit")->findBlock("upper")->getAttribute("slope", _flam_up_m);
    db->findBlock("flammability_limit")->findBlock("upper")->getAttribute("intercept",_flam_up_b);

    _use_flam_limits = true; 

  } 

  //Bullet proofing: 
  if ( _use_T_clip && _use_flam_limits ){ 
    throw ProblemSetupException( "Error: Cannot use temperature clip and flammability limits for the same westbrook/dryer source term.", __FILE__, __LINE__);
  } else if ( !_use_flam_limits && !_use_T_clip ){ 
    throw ProblemSetupException( "Error: Must use temperature clip OR flammability limit for the westbrook/dryer source term.", __FILE__, __LINE__);
  } 

  // add for table lookup
  _field_labels->add_species( d_o2_label ); 
  _field_labels->add_species( d_rho_label ); 
  _field_labels->add_species( d_T_label ); 

  _hot_spot = false; 
  if ( db->findBlock("hot_spot") ){
    ProblemSpecP db_hotspot = db->findBlock("hot_spot"); 
    ProblemSpecP the_geometry = db_hotspot->findBlock("geom_object");
    GeometryPieceFactory::create( the_geometry, _geom_hot_spot ); 
    db_hotspot->require("start_time",_start_time_hot_spot);
    db_hotspot->require("stop_time", _stop_time_hot_spot); 
    db_hotspot->require("temperature", _T_hot_spot);
    _hot_spot = true; 
  }

  // hard set some values...may want to change some of these to be inputs
  d_MW_O2 = 32.0; 
  d_R     = 8.314472; 
  d_Press = 101325; 

  d_MW_HC = 12.0*d_X + 1.0*d_Y; // compute the molecular weight from input information

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
WestbrookDryer::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "WestbrookDryer::eval";
  Task* tsk = scinew Task(taskname, this, &WestbrookDryer::computeSource, timeSubStep);

  _temperatureLabel   = VarLabel::find( d_T_label );
  _denLabel           = VarLabel::find( d_rho_label );
  _CstarMassFracLabel = VarLabel::find( d_cstar_label );
  if ( _CstarMassFracLabel == 0 ){ 
    throw ProblemSetupException( "Error: Could not locate the C* mass fraction label.", __FILE__, __LINE__);
  } 
  _CEqMassFracLabel   = VarLabel::find( d_ceq_label );
  _O2MassFracLabel    = VarLabel::find( d_o2_label ); 
  if ( _use_flam_limits && !_const_diluent ){ 
    _diluentLabel       = VarLabel::find( _diluent_label_name ); 
  }



  if (timeSubStep == 0 ) {

    tsk->computes(_src_label);
    tsk->computes(d_WDstrippingLabel); 
    tsk->computes(d_WDextentLabel); 

    tsk->requires( Task::OldDW, _temperatureLabel, Ghost::None, 0 ); 
    tsk->requires( Task::OldDW, _CstarMassFracLabel,  Ghost::None, 0 ); 
    tsk->requires( Task::OldDW, _CEqMassFracLabel, Ghost::None, 0 ); 
    tsk->requires( Task::OldDW, _denLabel,         Ghost::None, 0 ); 
    tsk->requires( Task::OldDW, _O2MassFracLabel,  Ghost::None, 0 ); 
    if ( _use_flam_limits && !_const_diluent ){ 
      tsk->requires( Task::OldDW, _diluentLabel, Ghost::None, 0 ); 
    } 

  } else {

    tsk->modifies(_src_label); 
    tsk->modifies(d_WDstrippingLabel); 
    tsk->modifies(d_WDextentLabel);   

    tsk->requires( Task::NewDW, _temperatureLabel, Ghost::None, 0 ); 
    tsk->requires( Task::NewDW, _CstarMassFracLabel,  Ghost::None, 0 ); 
    tsk->requires( Task::NewDW, _denLabel,         Ghost::None, 0 ); 
    tsk->requires( Task::NewDW, _CEqMassFracLabel, Ghost::None, 0 ); 
    tsk->requires( Task::NewDW, _O2MassFracLabel,  Ghost::None, 0 ); 
    if ( _use_flam_limits && !_const_diluent ){ 
      tsk->requires( Task::NewDW, _diluentLabel, Ghost::None, 0 ); 
    } 

  }

  tsk->requires( Task::OldDW, _field_labels->d_sharedState->get_delt_label(), Ghost::None, 0);

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
WestbrookDryer::computeSource( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 
    Vector Dx = patch->dCell(); 
    double vol = Dx.x();
#ifdef YDIM
    vol *= Dx.y();
#endif
#ifdef ZDIM
    vol *= Dx.z();
#endif
    

    CCVariable<double> CxHyRate; // rate source term  
    CCVariable<double> S; // stripping fraction 
    CCVariable<double> E; // extent of reaction 
    constCCVariable<double> T;      // temperature 
    constCCVariable<double> den;    // mixture density
    constCCVariable<double> Cstar;  // mass fraction of the hydrocarbon
    constCCVariable<double> Ceq;    // mass fraction of hydrocarbon for equilibrium
    constCCVariable<double> O2;     // O2 mass fraction
    constCCVariable<double> diluent; // Diluent mass fraction
    
    if ( new_dw->exists(_src_label, matlIndex, patch ) ){
      new_dw->getModifiable( CxHyRate , _src_label         , matlIndex , patch );
      new_dw->getModifiable( S        , d_WDstrippingLabel , matlIndex , patch );
      new_dw->getModifiable( E        , d_WDextentLabel    , matlIndex , patch );
      CxHyRate.initialize(0.0);
      S.initialize(0.0);
      E.initialize(0.0); 
      
      new_dw->get( T     , _temperatureLabel   , matlIndex , patch , Ghost::None , 0 );
      new_dw->get( Cstar , _CstarMassFracLabel , matlIndex , patch , Ghost::None , 0 );
      new_dw->get( den   , _denLabel           , matlIndex , patch , Ghost::None , 0 );
      new_dw->get( Ceq   , _CEqMassFracLabel   , matlIndex , patch , Ghost::None , 0 );
      new_dw->get( O2    , _O2MassFracLabel    , matlIndex , patch , Ghost::None , 0 ); 
      if ( _use_flam_limits && !_const_diluent ){ 
        new_dw->get( diluent, _diluentLabel,     matlIndex , patch , Ghost::None, 0 ); 
      } 

    } else {

      new_dw->allocateAndPut( CxHyRate , _src_label         , matlIndex , patch );
      new_dw->allocateAndPut( S        , d_WDstrippingLabel , matlIndex , patch );
      new_dw->allocateAndPut( E        , d_WDextentLabel    , matlIndex , patch );
    
      CxHyRate.initialize(0.0);
      S.initialize(0.0); 
      E.initialize(0.0); 

      old_dw->get( T     , _temperatureLabel   , matlIndex , patch , Ghost::None , 0 );
      old_dw->get( den   , _denLabel           , matlIndex , patch , Ghost::None , 0 );
      old_dw->get( Cstar , _CstarMassFracLabel , matlIndex , patch , Ghost::None , 0 );
      old_dw->get( Ceq   , _CEqMassFracLabel   , matlIndex , patch , Ghost::None , 0 );
      old_dw->get( O2    , _O2MassFracLabel    , matlIndex , patch , Ghost::None , 0 ); 
      if ( _use_flam_limits && !_const_diluent ){ 
        old_dw->get( diluent, _diluentLabel,     matlIndex , patch , Ghost::None, 0 ); 
      } 

    } 

    delt_vartype DT; 
    old_dw->get(DT, _field_labels->d_sharedState->get_delt_label()); 
    double dt = DT;
    Box patchInteriorBox = patch->getBox(); 

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      double f = Ceq[c] + Cstar[c];
      
      // Step 1: Compute stripping fraction and extent 
      double tiny = 1.0e-16;
      S[c] = 0.0; 
      double hc_wo_rxn = f * d_MF_HC_f1;

      if ( Cstar[c] > tiny ) 
        S[c] = Cstar[c] / hc_wo_rxn; 

      E[c] = 1.0 - S[c]; 

      // Step 2: Compute rate
      double rate = 0.0;
      if ( _use_T_clip ){ 

        double fake_diluent = 0.0; 
        rate = getRate( T[c], Cstar[c], O2[c], fake_diluent, f, den[c], dt, vol ); 

      } else { 

        if ( _const_diluent ){ 
          rate = getRate( _T_hot_spot, Cstar[c], O2[c], _const_diluent_mass_fraction, f, den[c], dt, vol ); 
        } else { 
          rate = getRate( _T_hot_spot, Cstar[c], O2[c], diluent[c], f, den[c], dt, vol ); 
        } 

      } 

      // Overwrite with hot spot if specified -- like a pilot light
      if ( _hot_spot ) { 

        double total_time = _field_labels->d_sharedState->getElapsedTime();

        for (std::vector<GeometryPieceP>::iterator giter = _geom_hot_spot.begin(); giter != _geom_hot_spot.end(); giter++){

          GeometryPieceP g_piece = *giter; 
          Box geomBox        = g_piece->getBoundingBox(); 
          Box intersectedBox = geomBox.intersect( patchInteriorBox ); 
          if ( !(intersectedBox.degenerate())) { 

            Point P = patch->cellPosition( c ); 
            
            if ( g_piece->inside(P) && total_time > _start_time_hot_spot && total_time < _stop_time_hot_spot ){ 

              if ( _use_T_clip ){ 
                double fake_diluent = 0.0; 
                rate = getRate( _T_hot_spot, Cstar[c], O2[c], fake_diluent, f, den[c], dt, vol ); 
              } else { 
                if ( _const_diluent ){ 
                  rate = getRate( _T_hot_spot, Cstar[c], O2[c], _const_diluent_mass_fraction, f, den[c], dt, vol ); 
                } else { 
                  rate = getRate( _T_hot_spot, Cstar[c], O2[c], diluent[c], f, den[c], dt, vol ); 
                } 
              } 

            }
          }
        }

        if ( total_time > _stop_time_hot_spot ){ 
          _hot_spot = false; 
        } 
      }

      CxHyRate[c] = rate; 

    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
WestbrookDryer::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "WestbrookDryer::initialize"; 

  Task* tsk = scinew Task(taskname, this, &WestbrookDryer::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter, _shared_state->allArchesMaterials()->getUnion()); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
WestbrookDryer::initialize( const ProcessorGroup* pc, 
                            const PatchSubset* patches, 
                            const MaterialSubset* matls, 
                            DataWarehouse* old_dw, 
                            DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> src;
    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      const VarLabel* tempVL = *iter; 
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, tempVL, matlIndex, patch ); 
      tempVar.initialize(0.0); 
    }
  }
}



