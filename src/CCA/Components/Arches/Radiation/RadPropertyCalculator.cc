#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Ports/Scheduler.h>
#ifdef HAVE_RADPROPS
#  include <radprops/AbsCoeffGas.h>
#  include <radprops/RadiativeSpecies.h>
#  include <radprops/Particles.h>
#endif

#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/Radiation/fortran/hottel_fort.h>

using namespace std; 
using namespace Uintah;

RadPropertyCalculator::RadPropertyCalculator( const int matl_index ) :
_matl_index( matl_index )
{}

RadPropertyCalculator::~RadPropertyCalculator(){
  for ( CalculatorVec::iterator i = _all_calculators.begin(); i != _all_calculators.end(); i++ ){ 
    delete *i;
  }
}

void
RadPropertyCalculator::problemSetup( const ProblemSpecP& db ){ 

  for ( ProblemSpecP db_pc = db->findBlock("calculator");
        db_pc != 0; db_pc = db_pc->findNextBlock("calculator") ) {

    RadPropertyCalculator::PropertyCalculatorBase* calculator;

    std::string calculator_type; 
    db_pc->getAttribute("type", calculator_type); 

    if ( calculator_type == "constant" ){ 
      calculator = scinew ConstantProperties(); 
    } else if ( calculator_type == "burns_christon" ){ 
      calculator = scinew BurnsChriston(); 
    } else if ( calculator_type == "hottel_sarofim"){
      calculator = scinew HottelSarofim(); 
    } else if ( calculator_type == "radprops" ){
#ifdef HAVE_RADPROPS
      calculator = scinew RadPropsInterface(); 
#else
      throw InvalidValue("Error: You haven't configured with the RadProps library (try configuring with --enable-wasatch_3p and --with-boost=DIR.)",__FILE__,__LINE__);
#endif
     } else { 
       throw InvalidValue("Error: Property calculator not recognized.",__FILE__, __LINE__); 
     } 

     if ( db_pc->findBlock("temperature")){ 
       db_pc->findBlock("temperature")->getAttribute("label", _temperature_name); 
     } else { 
       _temperature_name = "temperature"; 
     }

     bool complete; 
     complete = calculator->problemSetup( db_pc );

     if ( complete )
       _all_calculators.push_back(calculator); 
     else 
       throw InvalidValue("Error: Unable to setup radiation property calculator: "+calculator_type,__FILE__, __LINE__); 


  } 
}

//void 
//RadPropertyCalculator::sched_compute_radiation_properties( const LevelP& level, SchedulerP& sched, 
//                                                           const MaterialSet* matls, const int time_substep, 
//                                                           const bool doing_initialization )
//{
//
//  std::string taskname = "RadPropertyCalculator::compute_radiation_properties"; 
//  Task* tsk = scinew Task(taskname, this, &RadPropertyCalculator::compute_radiation_properties, 
//      time_substep, doing_initialization ); 
//
//  for ( CalculatorVec::iterator i = _all_calculators.begin(); i != _all_calculators.end(); i++ ){ 
//
//    const bool local_abskp = (*i)->has_abskp_local();  
//    const bool use_abskp   = (*i)->use_abskp(); 
//
//    if ( time_substep == 0 && !doing_initialization ){ 
//
//      tsk->computes( (*i)->get_abskg_label() );
//
//      if ( use_abskp && local_abskp ){ 
//        tsk->computes( (*i)->get_abskp_label() ); 
//      } else if ( use_abskp && !local_abskp ){ 
//        tsk->requires( Task::OldDW, (*i)->get_abskp_label(), Ghost::None, 0 );
//      }
//
//      //participating species from property calculator
//      std::vector<std::string> part_sp = (*i)->get_sp(); 
//
//      for ( std::vector<std::string>::iterator iter = part_sp.begin(); iter != part_sp.end(); iter++){
//        const VarLabel* label = VarLabel::find(*iter);
//        if ( label != 0 ){ 
//          tsk->requires( Task::OldDW, label, Ghost::None, 0 ); 
//        } else { 
//          throw ProblemSetupException("Error: Could not match species with varlabel: "+*iter,__FILE__, __LINE__);
//        }
//      }
//
//    } else if ( time_substep == 0 && doing_initialization ) {
//      tsk->computes( (*i)->get_abskg_label() );
//
//      if ( use_abskp && local_abskp ){ 
//        tsk->computes( (*i)->get_abskp_label() ); 
//      } else if ( use_abskp && !local_abskp ){ 
//        std::cout << " ABSKP LABEL=" << *((*i)->get_abskp_label()) << std::endl;
//        tsk->requires( Task::NewDW, (*i)->get_abskp_label(), Ghost::None, 0 );
//      }
//      //participating species from property calculator
//      std::vector<std::string> part_sp = (*i)->get_sp(); 
//
//      for ( std::vector<std::string>::iterator iter = part_sp.begin(); iter != part_sp.end(); iter++){
//        const VarLabel* label = VarLabel::find(*iter);
//        if ( label != 0 ){ 
//          tsk->requires( Task::NewDW, label, Ghost::None, 0 ); 
//        } else { 
//          throw ProblemSetupException("Error: Could not match species with varlabel: "+*iter,__FILE__, __LINE__);
//        }
//      }
//    } else { 
//
//      tsk->modifies( (*i)->get_abskg_label() );
//
//      if ( use_abskp && local_abskp ){ 
//        tsk->modifies( (*i)->get_abskp_label() ); 
//      } else if ( use_abskp && !local_abskp ){ 
//        tsk->requires( Task::NewDW, (*i)->get_abskp_label(), Ghost::None, 0 );
//      }
//      //participating species from property calculator
//      std::vector<std::string> part_sp = (*i)->get_sp(); 
//
//      for ( std::vector<std::string>::iterator iter = part_sp.begin(); iter != part_sp.end(); iter++){
//        const VarLabel* label = VarLabel::find(*iter);
//        if ( label != 0 ){ 
//          tsk->requires( Task::NewDW, label, Ghost::None, 0 ); 
//        } else { 
//          throw ProblemSetupException("Error: Could not match species with varlabel: "+*iter,__FILE__, __LINE__);
//        }
//      }
//
//    }
//  }
//
//  tsk->requires( Task::NewDW, VarLabel::find("volFraction"), Ghost::None, 0 ); 
//
//  _temperature_label = VarLabel::find(_temperature_name); 
//  if ( _temperature_label != 0 ){ 
//    if ( time_substep == 0 && !doing_initialization ){ 
//      tsk->requires( Task::OldDW, VarLabel::find(_temperature_name), Ghost::None, 0);
//    } else if ( time_substep == 0 && doing_initialization ){ 
//      tsk->requires( Task::NewDW, VarLabel::find(_temperature_name), Ghost::None, 0);
//    } else { 
//      tsk->requires( Task::NewDW, VarLabel::find(_temperature_name), Ghost::None, 0);
//    }
//  } else { 
//    throw ProblemSetupException("Error: Could not find the temperature label",__FILE__, __LINE__);
//  }
//
//  sched->addTask(tsk, level->eachPatch(), matls); 
//
//}
//
//
//void 
//RadPropertyCalculator::compute_radiation_properties( const ProcessorGroup* pc, 
//                                                     const PatchSubset* patches, 
//                                                     const MaterialSubset* matls, 
//                                                     DataWarehouse* old_dw, 
//                                                     DataWarehouse* new_dw, 
//                                                     const int time_substep, 
//                                                     const bool doing_initialization )
//{
//
//  //patch loop
//  for (int p=0; p < patches->size(); p++){
//    const Patch* patch = patches->get(p);
//    
//    //get other variables
//    constCCVariable<double> vol_fraction; 
//    constCCVariable<double> temperature; 
//
//    new_dw->get( vol_fraction, VarLabel::find("volFraction"), _matl_index, patch, Ghost::None, 0 ); 
//
//    for ( CalculatorVec::iterator i = _all_calculators.begin(); i != _all_calculators.end(); i++ ){ 
//     
//      DataWarehouse* which_dw; 
//    
//      const bool local_abskp = (*i)->has_abskp_local();  
//      const bool use_abskp   = (*i)->use_abskp(); 
//
//      CCVariable<double> abskg; 
//      CCVariable<double> abskp; 
//      constCCVariable<double> const_abskp; 
//
//      if ( time_substep == 0 && !doing_initialization ) { 
//        which_dw = old_dw; 
//        new_dw->allocateAndPut( abskg, (*i)->get_abskg_label(), _matl_index, patch );
//        if ( use_abskp && local_abskp ){ 
//          new_dw->allocateAndPut( abskp, (*i)->get_abskp_label(), _matl_index, patch );
//        } else if ( use_abskp && !local_abskp ){ 
//          old_dw->get( const_abskp, (*i)->get_abskp_label(), _matl_index, patch, Ghost::None, 0 ); 
//        }
//        old_dw->get( temperature, _temperature_label, _matl_index, patch, Ghost::None, 0 ); 
//      } else if ( time_substep == 0 && doing_initialization ){
//        which_dw = new_dw; 
//        new_dw->allocateAndPut( abskg, (*i)->get_abskg_label(), _matl_index, patch );
//        if ( use_abskp && local_abskp ){ 
//          new_dw->allocateAndPut( abskp, (*i)->get_abskp_label(), _matl_index, patch );
//        } else if ( use_abskp && !local_abskp ){ 
//          new_dw->get( const_abskp, (*i)->get_abskp_label(), _matl_index, patch, Ghost::None, 0 ); 
//        }
//        new_dw->get( temperature, _temperature_label, _matl_index, patch, Ghost::None, 0 ); 
//
//      } else { 
//        which_dw = new_dw; 
//        new_dw->getModifiable( abskg, (*i)->get_abskg_label(), _matl_index, patch );
//        if ( use_abskp && local_abskp ){ 
//          new_dw->getModifiable( abskp, (*i)->get_abskp_label(), _matl_index, patch );
//        } else if ( use_abskp && !local_abskp ){ 
//          new_dw->get( const_abskp, (*i)->get_abskp_label(), _matl_index, patch, Ghost::None, 0 ); 
//        }
//        new_dw->get( temperature, _temperature_label, _matl_index, patch, Ghost::None, 0 ); 
//      }
//
//      //participating species from property calculator
//      typedef std::vector<constCCVariable<double> > CCCV; 
//      CCCV species; 
//      std::vector<std::string> part_sp = (*i)->get_sp(); 
//
//      for ( std::vector<std::string>::iterator iter = part_sp.begin(); iter != part_sp.end(); iter++){
//        const VarLabel* label = VarLabel::find(*iter);
//        constCCVariable<double> spec; 
//        which_dw->get( spec, label, _matl_index, patch, Ghost::None, 0 ); 
//        species.push_back(spec); 
//      }
//
//      //initializing properties here.  This needs to be made consistent with BCs
//      if ( time_substep == 0 ){ 
//        abskg.initialize(1.0); //so that walls, bcs, etc, are fulling absorbing 
//        if ( use_abskp && local_abskp ){
//          abskp.initialize(0.0); 
//        }
//      }
//
//      //actually compute the properties
//      if ( time_substep == 0 )
//        (*i)->computeProps( patch, vol_fraction, species, temperature, abskg ); 
//
//      //sum in the particle contribution if needed
//      if ( use_abskp ){ 
//
//        if ( local_abskp ){ 
//          (*i)->sum_abs( abskg, abskp, patch ); 
//        } else { 
//          (*i)->sum_abs( abskg, const_abskp, patch ); 
//        }
//
//      }
//
//    } //calculator loop
//  }   //patch loop
//}

//--------------------------------------------------
// Below find the individual calculators
//--------------------------------------------------

//--------------------------------------------------
// Constant Properties
//--------------------------------------------------
RadPropertyCalculator::ConstantProperties::ConstantProperties() {
  _local_abskp = false; 
  _use_abskp = false; 
};
RadPropertyCalculator::ConstantProperties::~ConstantProperties() {};
    
bool RadPropertyCalculator::ConstantProperties::problemSetup( const ProblemSpecP& db ) {
        
  ProblemSpecP db_prop = db; 
  db_prop->getWithDefault("abskg_value",_abskg_value,0.0); 

  if ( db_prop->findBlock("abskg") ){ 
    db_prop->findBlock("abskg")->getAttribute("label",_abskg_name); 
  } else { 
    _abskg_name = "abskg"; 
  }

  //------------ check to see if scattering is turned on --//
  std::string radiation_model;
  db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->getAttribute("type",radiation_model) ; 

  if (radiation_model == "do_radiation"){
    db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->findBlock("DORadiationModel")->getWithDefault("ScatteringOn" ,_use_scatkt,false) ; 
  }
  else if ( radiation_model == "rmcrt_radiation"){
    db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->findBlock("RMCRT")->getWithDefault("ScatteringOn" ,_use_scatkt,false) ; 
  }
  //-------------------------------------------------------//

  //Create the particle absorption coeff as a <PropertyModel>

  const VarLabel* test_label = VarLabel::find(_abskg_name); 
  if ( test_label == 0 ){ 
    _abskg_label = VarLabel::create(_abskg_name, CCVariable<double>::getTypeDescription() ); 
  } else { 
    throw ProblemSetupException("Error: Abskg label already used for constant properties: "+_abskg_name,__FILE__, __LINE__);
  }

  bool property_on = true; 

  return property_on; 
}


    
void RadPropertyCalculator::ConstantProperties::compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, 
                                                              RadCalcSpeciesList species, constCCVariable<double>& mixT, 
                                                              CCVariable<double>& abskg ){ 
  abskg.initialize(_abskg_value); 

}

void RadPropertyCalculator::ConstantProperties::compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                    double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                    const int Nqn,   CCVariable<double>& abskp ){

  throw InvalidValue( "Error: No particle properties implemented for constant radiation properties.",__FILE__,__LINE__);

}

void RadPropertyCalculator::ConstantProperties::compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                    double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                    const int Nqn,   CCVariable<double>& scatkt ){

  throw InvalidValue( "Error: No particle properties implemented for constant radiation properties.",__FILE__,__LINE__);

}

//--------------------------------------------------
// Burns/Christon Properties
//--------------------------------------------------
RadPropertyCalculator::BurnsChriston::BurnsChriston() {
  _notSetMin = Point(SHRT_MAX, SHRT_MAX, SHRT_MAX);
  _notSetMax = Point(SHRT_MIN, SHRT_MIN, SHRT_MIN);
  _local_abskp = false; 
  _use_abskp = false; 
}

RadPropertyCalculator::BurnsChriston::~BurnsChriston() {}

bool RadPropertyCalculator::BurnsChriston::problemSetup( const ProblemSpecP& db ) { 

  ProblemSpecP db_prop = db;
  db_prop->getWithDefault("min", _min, _notSetMin);  // optional
  db_prop->getWithDefault("max", _max, _notSetMax);
  
  // bulletproofing  min & max must be set
  if( ( _min == _notSetMin && _max != _notSetMax) ||
      ( _min != _notSetMin && _max == _notSetMax) ){
    ostringstream warn;
    warn << "\nERROR:<property_calculator type=burns_christon>\n "
         << "You must specify both a min: "<< _min << " & max point: "<< _max <<"."; 
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  if ( db_prop->findBlock("abskg") ){ 
    db_prop->findBlock("abskg")->getAttribute("label",_abskg_name); 
  } else { 
    _abskg_name = "abskg"; 
  }

  //------------ check to see if scattering is turned on --//
  std::string radiation_model;
  db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->getAttribute("type",radiation_model) ; 

  if (radiation_model == "do_radiation"){
    db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->findBlock("DORadiationModel")->getWithDefault("ScatteringOn" ,_use_scatkt,false) ; 
  }
  else if ( radiation_model == "rmcrt_radiation"){
    db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->findBlock("RMCRT")->getWithDefault("ScatteringOn" ,_use_scatkt,false) ; 
  }
  //-------------------------------------------------------//

  //no abskp
  _abskp_name = "NA"; 

  const VarLabel* test_label = VarLabel::find(_abskg_name); 
  if ( test_label == 0 ){ 
    _abskg_label = VarLabel::create(_abskg_name, CCVariable<double>::getTypeDescription() ); 
  } else { 
    throw ProblemSetupException("Error: Abskg label already used for constant properties: "+_abskg_name,__FILE__, __LINE__);
  }

  bool property_on = true; 
  return property_on; 
}

void RadPropertyCalculator::BurnsChriston::compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg ){ 
  
  BBox domain(_min,_max);
  
  // if the user didn't specify the min and max 
  // use the grid's domain
  if( _min == _notSetMin  ||  _max == _notSetMax ){
    const Level* level = patch->getLevel();
    GridP grid  = level->getGrid();
    grid->getInteriorSpatialRange(domain);
    _min = domain.min();
    _max = domain.max();
  }
  
  Point midPt((_max - _min)/2 + _min);
  
  for (CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter){ 
    IntVector c = *iter; 
    Point pos = patch->getCellPosition(c);
    
    if(domain.inside(pos)){
      abskg[c] = 0.90 * ( 1.0 - 2.0 * fabs( pos.x() - midPt.x() ) )
                      * ( 1.0 - 2.0 * fabs( pos.y() - midPt.y() ) )
                      * ( 1.0 - 2.0 * fabs( pos.z() - midPt.z() ) ) 
                      + 0.1;
    }
  } 
}

void RadPropertyCalculator::BurnsChriston::compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                    double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                    const int Nqn,   CCVariable<double>& abskp ){

  throw InvalidValue( "Error: No particle properties implemented for Burns/Christon radiation properties.",__FILE__,__LINE__);
}

void RadPropertyCalculator::BurnsChriston::compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                    double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                    const int Nqn,   CCVariable<double>& scatkt ){

  throw InvalidValue( "Error: No particle properties implemented for Burns/Christon radiation properties.",__FILE__,__LINE__);
}

/// --------------------------------------
//  Hottel/Sarofim 
// ---------------------------------------
RadPropertyCalculator::HottelSarofim::HottelSarofim() {
  _local_abskp = false; 
  _use_abskp = false; 
};
RadPropertyCalculator::HottelSarofim::~HottelSarofim() {};
    
bool 
RadPropertyCalculator::HottelSarofim::problemSetup( const ProblemSpecP& db ) {
  ProblemSpecP db_h = db;
  db_h->require("opl",d_opl);
  _co2_name = "CO2"; 
  if ( db_h->findBlock("co2")){ 
    db_h->findBlock("co2")->getAttribute("label",_co2_name);  
  }
  _h2o_name = "H2O"; 
  if ( db_h->findBlock("h2o")){ 
    db_h->findBlock("h2o")->getAttribute("label",_h2o_name);  
  }
  _soot_name = "soot"; 
  if ( db_h->findBlock("soot")){ 
    db_h->findBlock("soot")->getAttribute("label",_soot_name);  
  }
  
  if ( db_h->findBlock("abskg") ){ 
    db_h->findBlock("abskg")->getAttribute("label",_abskg_name); 
  } else { 
    _abskg_name = "abskg"; 
  }

  const VarLabel* test_label = VarLabel::find(_abskg_name); 
  if ( test_label == 0 ){ 
    _abskg_label = VarLabel::create(_abskg_name, CCVariable<double>::getTypeDescription() ); 
  } else { 
    throw ProblemSetupException("Error: Abskg label already used for constant properties: "+_abskg_name,__FILE__, __LINE__);
  }

  if ( db_h->findBlock("abskp")){ 
    db_h->findBlock("abskp")->getAttribute("label",_abskp_name); 
    _use_abskp = true; 
  }

  //------------ check to see if scattering is turned on --//
  std::string radiation_model;
  db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->getAttribute("type",radiation_model) ; 

  if (radiation_model == "do_radiation"){
    db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->findBlock("DORadiationModel")->getWithDefault("ScatteringOn" ,_use_scatkt,false) ; 
  }
  else if ( radiation_model == "rmcrt_radiation"){
    db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->findBlock("RMCRT")->getWithDefault("ScatteringOn" ,_use_scatkt,false) ; 
  }
  //-------------------------------------------------------//

  bool property_on = true;
  return property_on; 

}

void 
RadPropertyCalculator::HottelSarofim::compute_abskg( const Patch* patch, 
    constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, 
    constCCVariable<double>& mixT, CCVariable<double>& abskg ){ 

  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

  fort_hottel(idxLo, idxHi, mixT,
              species[0], species[1], VolFractionBC,
              d_opl, species[2], abskg);

}

void 
RadPropertyCalculator::HottelSarofim::compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                    double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                    const int Nqn,   CCVariable<double>& abskp ){

  throw InvalidValue( "Error: No particle properties implemented for Hottel-Sarofim radiation properties.",__FILE__,__LINE__);

}

void 
RadPropertyCalculator::HottelSarofim::compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                    double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                    const int Nqn,   CCVariable<double>& scatkt ){

  throw InvalidValue( "Error: No particle properties implemented for Hottel-Sarofim radiation properties.",__FILE__,__LINE__);

}

vector<std::string> 
RadPropertyCalculator::HottelSarofim::get_sp(){

  _the_species.clear(); 
  _the_species.push_back(_co2_name);
  _the_species.push_back(_h2o_name);
  _the_species.push_back(_soot_name);
   return _the_species;

}

bool 
RadPropertyCalculator::HottelSarofim::does_scattering(){ return false; } 

/// --------------------------------------
//  RADPROPS
// ---------------------------------------
#ifdef HAVE_RADPROPS
RadPropertyCalculator::RadPropsInterface::RadPropsInterface() 
{
  _gg_radprops   = 0;
  _part_radprops = 0; 
  _p_ros_abskp  = false; 
  _p_planck_abskp = false; 
  _local_abskp = false; 
  _use_abskp = false; 
}

RadPropertyCalculator::RadPropsInterface::~RadPropsInterface() {
    
  if ( _gg_radprops != 0 ) 
    delete _gg_radprops; 

  if ( _part_radprops != 0 ) 
    delete _part_radprops; 

}
    
bool RadPropertyCalculator::RadPropsInterface::problemSetup( const ProblemSpecP& db ) {

  if ( db->findBlock( "grey_gas" ) ){

    ProblemSpecP db_gg = db->findBlock( "grey_gas" );

    db_gg->getWithDefault("mix_mol_w_label",_mix_mol_weight_name,"mixture_molecular_weight"); 
    std::string inputfile;
    db_gg->require("inputfile",inputfile); 

    //allocate gray gas object: 
    _gg_radprops = scinew GreyGas( inputfile ); 

    //get list of species: 
    _radprops_species = _gg_radprops->species();

    // mixture molecular weight will always be the first entry 
    // Note that we will assume the table value is the inverse
    _species.insert(_species.begin(), _mix_mol_weight_name);

    // NOTE: this requires that the table names match the RadProps name.  This is, in general, a pretty 
    // bad assumption.  Need to make this more robust later on...
    int total_sp = _radprops_species.size(); 
    for ( int i = 0; i < total_sp; i++ ){ 
      std::string which_species = species_name( _radprops_species[i] ); 
      _species.push_back( which_species ); 

      //entering the inverse here for convenience 
      if ( which_species == "CO2" ){ 
        _sp_mw.push_back(1.0/44.0);
      } else if ( which_species == "H2O" ){ 
        _sp_mw.push_back(1.0/18.0); 
      } else if ( which_species == "CO" ){ 
        _sp_mw.push_back(1.0/28.0); 
      } else if ( which_species == "NO" ){
        _sp_mw.push_back(1.0/30.0); 
      } else if ( which_species == "OH" ){
        _sp_mw.push_back(1.0/17.0); 
      } 
    } 

  }else { 

    throw InvalidValue( "Error: Only grey gas properties are available at this time.",__FILE__,__LINE__);

  }

  if ( db->findBlock("abskg") ){ 
    db->findBlock("abskg")->getAttribute("label",_abskg_name); 
  } else { 
    _abskg_name = "abskg"; 
  }

  if ( db->findBlock("abskp") ){ 
    db->findBlock("abskp")->getAttribute("label",_abskp_name); 
    _use_abskp = true; 
  }

  //------------ check to see if scattering is turned on --//
  std::string radiation_model;
  db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->getAttribute("type",radiation_model) ; 

  if (radiation_model == "do_radiation"){
    db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->findBlock("DORadiationModel")->getWithDefault("ScatteringOn" ,_use_scatkt,false) ; 
  }
  else if ( radiation_model == "rmcrt_radiation"){
    db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources")->findBlock("src")->findBlock("RMCRT")->getWithDefault("ScatteringOn" ,_use_scatkt,false) ; 
  }
  //-------------------------------------------------------//
  
  if (_use_scatkt){
    _scatkt_name = "scatkt";
      _scatkt_label = VarLabel::find(_scatkt_name); 
    if ( _scatkt_label == 0 ){ 
      throw ProblemSetupException("Error: scatkt label not found! This label should be created in the Radiation model!"+_scatkt_name,__FILE__, __LINE__);
    } 
  }


  const VarLabel* test_label = VarLabel::find(_abskg_name); 
  if ( test_label == 0 ){ 
    _abskg_label = VarLabel::create(_abskg_name, CCVariable<double>::getTypeDescription() ); 
  } else { 
    throw ProblemSetupException("Error: Abskg label already in use: "+_abskg_name,__FILE__, __LINE__);
  }

  // For particles: 
  _does_scattering = false; 
  if ( db->findBlock( "particles" ) ){ 

    ProblemSpecP db_p = db->findBlock( "particles" ); 

    double real_part = 0; 
    double imag_part = 0; 
    db_p->require( "complex_ir_real", real_part ); 
    db_p->require( "complex_ir_imag", imag_part ); 

    std::string which_model = "none"; 
    db_p->require( "model_type", which_model );
    if ( which_model == "planck" ){ 
      _p_planck_abskp = true; 
    } else if ( which_model == "rossland" ){ 
      _p_ros_abskp = true; 
    } else { 
      throw InvalidValue( "Error: Particle model not recognized for abskp.",__FILE__,__LINE__);
    }   

    std::complex<double> complex_ir( real_part, imag_part ); 

    _part_radprops = scinew ParticleRadCoeffs( complex_ir ); 

    _does_scattering = true; 
    _local_abskp = true; 

    _abskp_label = VarLabel::create(_abskp_name, CCVariable<double>::getTypeDescription() ); 

  }
  return true; 
  
}
    
void RadPropertyCalculator::RadPropsInterface::compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species,  constCCVariable<double>& mixT, CCVariable<double>& abskg)
{ 

  int N = species.size(); 

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter; 

    double plankCff = 0.0;
    double rossCff  = 0.0; 
    double effCff   = 0.0; 
    std::vector<double> mol_frac; 
    double T        = mixT[c];
    double VolFraction = VolFractionBC[c];

    //convert mass frac to mol frac
    for ( int i = 1; i < N; i++ ){ 
      // Note that the species molecular weights and mixture molecular weights 
      // are actually the inverse 
      // as set in the problemsetup function.
      // Also note that the mixture molecular weight arriving from the table
      // is assumed to be the inverse mixture molecular weight

      double value = (species[0])[c]==0 ? 0.0 : (species[i])[c] * _sp_mw[i-1] * 1.0 / (species[0])[c];
      //                                          ^^species^^^^    ^^1/MW^^^^^  ^^^^^MIX MW^^^^^^^^^^
      
      if ( value < 0 ){ 
        throw InvalidValue( "Error: For some reason I am getting negative mol fractions in the scattering portion of radiation property calculator.",__FILE__,__LINE__);
      } 

      mol_frac.push_back(value); 

    } 

    if ( VolFraction > 1.e-16 ){

      _gg_radprops->mixture_coeffs(plankCff, rossCff, effCff, mol_frac, T);

      abskg[c] = effCff*100; // from cm^-1 to m^-1 //need to generalize this to the other coefficients

    } else { 

      abskg[c] = 0.0; 

    }
  }
}

void RadPropertyCalculator::RadPropsInterface::compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                    double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                    const int Nqn, CCVariable<double>& abskp ){

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter; 

    double plankCff = 0.0;
    double rossCff  = 0.0; 
    double effCff   = 0.0; 
    double VolFraction = VolFractionBC[c];
    double unscaled_weight;
    double unscaled_size;


    if ( VolFraction > 1.e-16 ){


    
      //now compute the particle values: 
      abskp[c] = 0.0; 
      for ( int i = 0; i < Nqn; i++ ){ 

        unscaled_weight = (weights[i])[c]*weights_scaling_constant;
        unscaled_size = (size[i])[c]*size_scaling_constant/(weights[i])[c];

        if ( _p_planck_abskp ){ 

          double abskp_i = _part_radprops->planck_abs_coeff( unscaled_size, (pT[i])[c] );
          abskp[c] += abskp_i * unscaled_weight; 
          
        } else if ( _p_ros_abskp ){ 

          double abskp_i =  _part_radprops->ross_abs_coeff( unscaled_size, (pT[i])[c] );
          abskp[c] += abskp_i * unscaled_weight; 

        } 
      }


    }else{    

      abskp[c] = 0.0;

    }
  }
}


void RadPropertyCalculator::RadPropsInterface::compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                    double size_scaling_constant, RadCalcSpeciesList size, RadCalcSpeciesList pT, double weights_scaling_constant, RadCalcSpeciesList weights, 
                                    const int Nqn, CCVariable<double>& scatkt ){

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter; 

    double plankCff = 0.0;
    double rossCff  = 0.0; 
    double effCff   = 0.0; 
    double VolFraction = VolFractionBC[c];
    double unscaled_weight;
    double unscaled_size;


    if ( VolFraction > 1.e-16 ){


    
      //now compute the particle values: 
      scatkt[c] = 0.0; 
      for ( int i = 0; i < Nqn; i++ ){ 

        unscaled_weight = (weights[i])[c]*weights_scaling_constant;
        unscaled_size = (size[i])[c]*size_scaling_constant/(weights[i])[c];

        if ( _p_planck_abskp ){ 

          double scatkt_i = _part_radprops->planck_sca_coeff( unscaled_size, (pT[i])[c] );
          scatkt[c] += scatkt_i * unscaled_weight; 
          
        } else if ( _p_ros_abskp ){ 

          double scatkt_i =  _part_radprops->ross_sca_coeff( unscaled_size, (pT[i])[c] );
          scatkt[c] += scatkt_i * unscaled_weight; 

        } 
      }


    }else{    

      scatkt[c] = 0.0;

    }
  }
}

#endif 

