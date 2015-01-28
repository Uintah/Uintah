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


//--------------------------------------------------
// Below find the individual calculators
//--------------------------------------------------

//--------------------------------------------------
// Constant Properties
//--------------------------------------------------
RadPropertyCalculator::ConstantProperties::ConstantProperties() {

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



//--------------------------------------------------
// Burns/Christon Properties
//--------------------------------------------------
RadPropertyCalculator::BurnsChriston::BurnsChriston() {
  _notSetMin = Point(SHRT_MAX, SHRT_MAX, SHRT_MAX);
  _notSetMax = Point(SHRT_MIN, SHRT_MIN, SHRT_MIN);
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



/// --------------------------------------
//  Hottel/Sarofim 
// ---------------------------------------
RadPropertyCalculator::HottelSarofim::HottelSarofim() {

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



vector<std::string> 
RadPropertyCalculator::HottelSarofim::get_sp(){

  _the_species.clear(); 
  _the_species.push_back(_co2_name);
  _the_species.push_back(_h2o_name);
  _the_species.push_back(_soot_name);
   return _the_species;

}


/// --------------------------------------
//  RADPROPS
// ---------------------------------------
#ifdef HAVE_RADPROPS
RadPropertyCalculator::RadPropsInterface::RadPropsInterface() 
{
  _gg_radprops   = 0;
}

RadPropertyCalculator::RadPropsInterface::~RadPropsInterface() {
    
  if ( _gg_radprops != 0 ) 
    delete _gg_radprops; 
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


  


  const VarLabel* test_label = VarLabel::find(_abskg_name); 
  if ( test_label == 0 ){ 
    _abskg_label = VarLabel::create(_abskg_name, CCVariable<double>::getTypeDescription() ); 
  } else { 
    throw ProblemSetupException("Error: Abskg label already in use: "+_abskg_name,__FILE__, __LINE__);
  }

  return true; 
  
}
    
void RadPropertyCalculator::RadPropsInterface::compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species,  constCCVariable<double>& mixT, CCVariable<double>& abskg)
{ 

  int N = species.size(); 
  double plankCff = 0.0;
  double rossCff  = 0.0; 
  double effCff   = 0.0; 

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter; 

    std::vector<double> mol_frac; 
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

      _gg_radprops->mixture_coeffs(plankCff, rossCff, effCff, mol_frac, mixT[c]);

      abskg[c] = effCff*100.0; // from cm^-1 to m^-1 //need to generalize this to the other coefficients

    } else { 

   //   abskg[c] = 1.0; 

    }
  }
}




#endif 


RadPropertyCalculator::coalOptics::coalOptics(const ProblemSpecP& db, bool scatteringOn){

  _scatteringOn = scatteringOn; 



  construction_success=true;


  if ( db->findBlock("particles")->findBlock("abskp") ){ 
    db->findBlock("particles")->findBlock("abskp")->getAttribute("label",_abskp_name); 
  }else{
    throw ProblemSetupException("Error: abskp name not found! This should be specified in the input file!",__FILE__, __LINE__);
  }


  //------------------2 (or 3) components for coal ----------//
  _ncomp=2;
  vector < std::string > base_composition_names(_ncomp);
  base_composition_names[0] = "Charmass_";
  base_composition_names[1] = "RCmass_";
  //---------------------------------------------------------//
  
  //-----------------All class objects of this type should do this---------------------//
  db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->require( "number_quad_nodes", _nQn_part ); 
  if ( _nQn_part ==0){
    construction_success = false;
  }
  _abskp_label = VarLabel::create(_abskp_name, CCVariable<double>::getTypeDescription() ); 
  _abskp_label_vector = vector<const VarLabel* >(_nQn_part);
  for (int i=0; i< _nQn_part ; i++){
    std::stringstream out; 
    out << _abskp_name << "_" << i;
    _abskp_label_vector[i] = VarLabel::create(out.str(), CCVariable<double>::getTypeDescription() ); 
  }
  //-----------------------------------------------------------------------------------//
  
 ProblemSpecP db_coal=db->findBlock("particles")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties");
 
 if (db_coal == 0){
   throw ProblemSetupException("Error: Coal properties not found! Need Optical Coal properties!",__FILE__, __LINE__);
 }else if (db_coal->findBlock("optics")==0){
   throw ProblemSetupException("Error: Coal properties not found! Need Optical Coal properties!",__FILE__, __LINE__);
 }

  db_coal->findBlock("optics")->require( "RawCoal_real", _rawCoalReal ); 
  db_coal->findBlock("optics")->require( "RawCoal_imag", _rawCoalImag ); 
  db_coal->findBlock("optics")->require( "Ash_real", _ashReal ); 
  db_coal->findBlock("optics")->require( "Ash_imag", _ashImag ); 

  _charReal=_rawCoalReal; // assume char and RC have same optical props
  _charImag=_rawCoalImag; // assume char and RC have same optical props

  if (_rawCoalReal > _ashReal) {
    _HighComplex=std::complex<double> ( _rawCoalReal, _rawCoalImag );  
    _LowComplex=std::complex<double> ( _ashReal, _ashImag );  
  } else{
    _HighComplex=std::complex<double> ( _ashReal, _ashImag );  
    _LowComplex=std::complex<double>  (_rawCoalReal, _rawCoalImag );  
  }

  /// complex index of refraction for pure coal components   
  ///  asymmetry parameters for pure coal components
  _charAsymm=1.0;
  _rawCoalAsymm=1.0;
  _ashAsymm=-1.0;

  //
    _complexIndexReal_label = vector<const VarLabel* >(_nQn_part);

    for (int i=0; i<_nQn_part ; i++ ){
      std::stringstream out1; 
      out1 << "complexIndexReal_" << i;
      _complexIndexReal_label[i]= VarLabel::create(out1.str(), CCVariable<double>::getTypeDescription() ); 
    }

  if ( _scatteringOn){
    _scatkt_name = "scatkt";
    _asymmetryParam_name="asymmetryParam"; 
    _scatkt_label = VarLabel::create(_scatkt_name,CCVariable<double>::getTypeDescription()); 
    _asymmetryParam_label = VarLabel::create(_asymmetryParam_name,CCVariable<double>::getTypeDescription()); 
    if (_scatkt_label==0){
      throw ProblemSetupException("Error: scattering coefficient label not created!!!?",__FILE__, __LINE__);
    } 
  }

  _composition_names = std::vector < std::string > (_ncomp*_nQn_part);

  for (int j=0; j< _ncomp ; j++ ){
    for (int i=0; i< _nQn_part ; i++ ){
      std::stringstream out; 
      out << base_composition_names[j] << i;
      _composition_names[i+j*_nQn_part] =out.str(); 
    }
  }
  double density;
  db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties")->require( "density", density ); 

  vector<double>  particle_sizes ;        /// particle sizes in diameters
  db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties")->require( "diameter_distribution", particle_sizes ); 

  double ash_massfrac;  
  db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal")->findBlock("Properties")->findBlock("ultimate_analysis")->require("ASH", ash_massfrac); 

  _ash_mass = vector<double>(_nQn_part);        /// particle sizes in diameters

  for (int i=0; i< _nQn_part ; i++ ){
    _ash_mass[i] = pow(particle_sizes[i], 3.0)/6*M_PI*density*ash_massfrac;
  }




  _part_radprops = 0; 
  ProblemSpecP db_p = db->findBlock( "particles" ); 



  std::string which_model = "none"; 
  db_p->require( "model_type", which_model );
  if ( which_model == "planck" ){ 
    _p_planck_abskp = true; 
  } else if ( which_model == "rossland" ){ 
    _p_ros_abskp = true; 
  } else { 
    throw InvalidValue( "Error: Particle model not recognized for abskp.",__FILE__,__LINE__);
  }   

  _part_radprops = scinew ParticleRadCoeffs3D( _LowComplex, _HighComplex, 1e-4, 1e-7, 10, 3 );

  _computeComplexIndex = true; // complex index of refraction needed
}

RadPropertyCalculator::coalOptics::~coalOptics(){


  for (int i=0; i< _nQn_part ; i++ ){
    VarLabel::destroy(_complexIndexReal_label[i]); 
  }
  if ( _scatteringOn ) {
    VarLabel::destroy(_asymmetryParam_label); 
    VarLabel::destroy(_scatkt_label); 
  }
    delete _part_radprops; 
}


bool RadPropertyCalculator::coalOptics::problemSetup(Task* tsk,int time_substep){


  if (time_substep ==0) {  // only populate labels on first time step....otherwise you get duplicates!


    for (int i=0; (signed) _compositionLabels.size() < _ncomp*_nQn_part ; i++ ){  // unique loop criteria ensures no label duplicate.
      const VarLabel* temp =  VarLabel::find(_composition_names[i]);
      if (temp == 0){
        proc0cout << "Coal optical-props is unable to find label with name: "<<_composition_names[i] << " \n";
        throw ProblemSetupException("Error: Could not find the label"+_composition_names[i],__FILE__, __LINE__);
      }
      else
        _compositionLabels.push_back( temp);
    }

    for (int i=0; i< _ncomp*_nQn_part ; i++ ){
      tsk->requires( Task::NewDW,_compositionLabels[i],  Ghost::None, 0);
    }
    for (int i=0; i< _nQn_part ; i++ ){
      tsk->computes(  _complexIndexReal_label[i] );
    }
    if (_scatteringOn){
      tsk->computes(  _asymmetryParam_label   );
    }
  }
  else{
    for (int i=0; i< _ncomp*_nQn_part ; i++ ){
      tsk->requires( Task::NewDW,_compositionLabels[i] , Ghost::None, 0); // this should be new_dw, but i'm getting an error BEN?
    }
    for (int i=0; i< _nQn_part ; i++ ){
      tsk->modifies(  _complexIndexReal_label[i] );
    }
    if (_scatteringOn){
      tsk->modifies(  _asymmetryParam_label   );
    }
  }



  return true;
}

void RadPropertyCalculator::coalOptics::compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                                       RadCalcSpeciesList size, RadCalcSpeciesList pT, RadCalcSpeciesList weights, 
                                                       const int Nqn, CCVariable<double>& abskpt, 
                                                       SCIRun::StaticArray < CCVariable<double> >  &abskp,
                                                       SCIRun::StaticArray < CCVariable<double> >  &complexReal){

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter; 

    double VolFraction = VolFractionBC[c];
    double unscaled_weight;
    double unscaled_size;


    if ( VolFraction > 1.e-16 ){
      //now compute the particle values: 
      abskpt[c] = 0.0; 
      for ( int i = 0; i < Nqn; i++ ){ 

        unscaled_weight = (weights[i])[c];
        unscaled_size = (size[i])[c];

        if ( _p_planck_abskp ){ 

          abskp[i][c] = _part_radprops->planck_abs_coeff( unscaled_size, (pT[i])[c],complexReal[i][c] )* unscaled_weight;
          //double abskp_i = _part_radprops->planck_abs_coeff( unscaled_size, (pT[i])[c] );
          abskpt[c] += abskp[i][c] ; 
          
        } else if ( _p_ros_abskp ){ 

          abskp[i][c] = _part_radprops->ross_abs_coeff( unscaled_size, (pT[i])[c],complexReal[i][c] )* unscaled_weight;
          //double abskp_i = _part_rossprops->planck_abs_coeff( unscaled_size, (pT[i])[c] );
          abskpt[c] += abskp[i][c] ; 

        } 
      }


    }else{    

      abskpt[c] = 0.0;

    }
  }
}


void RadPropertyCalculator::coalOptics::compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                                        RadCalcSpeciesList size, RadCalcSpeciesList pT, RadCalcSpeciesList weights, 
                                                        const int Nqn, CCVariable<double>& scatkt,
                                                        SCIRun::StaticArray < CCVariable<double> > &scatktQuad,
                                                        SCIRun::StaticArray < CCVariable<double> >  &complexReal){

  for ( int i = 0; i < Nqn; i++ ){ 
    scatktQuad[i].allocate(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex());
    scatktQuad[i].initialize(0.0);
  }

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){


    IntVector c = *iter; 

    double VolFraction = VolFractionBC[c];
    double unscaled_weight;
    double unscaled_size;


    if ( VolFraction > 1.e-16 ){

      scatkt[c] = 0.0; 
      for ( int i = 0; i < Nqn; i++ ){ 

        unscaled_weight = (weights[i])[c];
        unscaled_size = (size[i])[c];
        if ( _p_planck_abskp ){ 
          double scatkt_i = _part_radprops->planck_sca_coeff( unscaled_size, (pT[i])[c], complexReal[i][c]);
          //double scatkt_i = _part_radprops->planck_sca_coeff( unscaled_size, (pT[i])[c]);
          scatktQuad[i][c]=scatkt_i* unscaled_weight;
          scatkt[c] += scatkt_i * unscaled_weight; 
          
        } else if ( _p_ros_abskp ){ 
          double scatkt_i =  _part_radprops->ross_sca_coeff( unscaled_size, (pT[i])[c] , complexReal[i][c]);
          //double scatkt_i =  _part_radprops->ross_sca_coeff( unscaled_size, (pT[i])[c] );
          scatktQuad[i][c]=scatkt_i* unscaled_weight;
          scatkt[c] += scatkt_i * unscaled_weight; 
        } 
      }
    }else{    

      scatkt[c] = 0.0;

    }
  }
}

void RadPropertyCalculator::coalOptics::computeComplexIndex( const Patch* patch,
                                                             constCCVariable<double>& VolFractionBC,
                                                             SCIRun::StaticArray < constCCVariable<double> > &composition,
                                                             SCIRun::StaticArray < CCVariable<double> >  &complexReal){

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter; 
    if ( VolFractionBC[c]> 1.e-16){
      //now compute the particle values: 
      for ( int i = 0; i <_nQn_part; i++ ){ 
       double total_mass =composition[i][c] + composition[i+_nQn_part][c] + _ash_mass[i];
       complexReal[i][c]  =  (composition[i][c]*_charReal+composition[i+_nQn_part][c]*_rawCoalReal+_ash_mass[i]*_ashReal)/total_mass;
      }
    }
    else{
    }// else, phase complex values remain zero in intrusions (set upstream).
  }

return;
}


void RadPropertyCalculator::coalOptics::computeAsymmetryFactor( const Patch* patch,
                                                                constCCVariable<double>& VolFractionBC,
                                                                SCIRun::StaticArray < CCVariable<double> > &scatktQuad, 
                                                                SCIRun::StaticArray < constCCVariable<double> > &composition,
                                                                CCVariable<double>  &scatkt,
                                                                CCVariable<double>  &asymmetryParam){

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter; 
    if ( VolFractionBC[c]> 1.e-16){

      for ( int i = 0; i <_nQn_part; i++ ){ 
       double total_mass =composition[i][c] + composition[i+_nQn_part][c] + _ash_mass[i];
      asymmetryParam[c] = (composition[i][c]*_charAsymm+composition[i+_nQn_part][c]*_rawCoalAsymm+_ash_mass[i]*_ashAsymm)/(total_mass*scatkt[c])*scatktQuad[i][c];
      }
      asymmetryParam[c] /= scatkt[c];
    } // else, phase function remain zero in intrusions (set upstream).
  }

return;
}

RadPropertyCalculator::constantCIF::constantCIF(const ProblemSpecP& db, bool scatteringOn){

  construction_success=true;
  _scatteringOn = scatteringOn; 


  if ( db->findBlock("particles")->findBlock("abskp") ){ 
    db->findBlock("particles")->findBlock("abskp")->getAttribute("label",_abskp_name); 
    double realCIF;
    double imagCIF;
    db->findBlock("particles")->require("complex_ir_real",realCIF); 
    db->findBlock("particles")->require("complex_ir_imag",imagCIF); 
    db->findBlock("particles")->getWithDefault("const_asymmFact",_constAsymmFact,0.0); 
    std::complex<double>  CIF(realCIF, imagCIF );  
    _part_radprops = scinew ParticleRadCoeffs(CIF);  
    std::string which_model = "none"; 
    db->findBlock("particles")->require("model_type", which_model);
    if ( which_model == "planck" ){ 
      _p_planck_abskp = true; 
    } else if ( which_model == "rossland" ){ 
      _p_ros_abskp = true; 
    } else { 
      throw InvalidValue( "Error: Particle model not recognized for abskp.",__FILE__,__LINE__);
    }   
  }else{
    throw ProblemSetupException("Error: abskp name not found! This should be specified in the input file!",__FILE__, __LINE__);
  }


  //-----------------All class objects of this type should do this---------------------//
  db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->require( "number_quad_nodes", _nQn_part ); 
  if ( _nQn_part ==0){
    construction_success = false;
  }
  _abskp_label = VarLabel::create(_abskp_name, CCVariable<double>::getTypeDescription() ); 
  _abskp_label_vector = vector<const VarLabel* >(_nQn_part);
  for (int i=0; i<_nQn_part; i++){
    std::stringstream out; 
    out << _abskp_name << "_" << i;
    _abskp_label_vector[i] = VarLabel::create(out.str(), CCVariable<double>::getTypeDescription() ); 
  }
  //-----------------------------------------------------------------------------------//
  
  if ( _scatteringOn){
    _scatkt_name = "scatkt";
    _asymmetryParam_name="asymmetryParam"; 
    _scatkt_label = VarLabel::create(_scatkt_name,CCVariable<double>::getTypeDescription()); 
    _asymmetryParam_label = VarLabel::create(_asymmetryParam_name,CCVariable<double>::getTypeDescription()); 
    if (_scatkt_label==0){
      throw ProblemSetupException("Error: scattering coefficient label not created!!!?",__FILE__, __LINE__);
    } 
  }
  _computeComplexIndex = false; // complex index of refraction not needed for this model
 
}

RadPropertyCalculator::constantCIF::~constantCIF(){
  if ( _scatteringOn ) {
    VarLabel::destroy(_asymmetryParam_label); 
    VarLabel::destroy(_scatkt_label); 
  }
    delete _part_radprops; 
}


bool RadPropertyCalculator::constantCIF::problemSetup(Task* tsk,int time_substep){

  if (time_substep ==0) { 
    if (_scatteringOn){
      tsk->computes(  _asymmetryParam_label   );
    }
  }
  else{
    if (_scatteringOn){
      tsk->modifies(  _asymmetryParam_label   );
    }
  }
  return true;
}

void RadPropertyCalculator::constantCIF::compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                                        RadCalcSpeciesList size, RadCalcSpeciesList pT, RadCalcSpeciesList weights, 
                                                        const int Nqn, CCVariable<double>& abskpt, 
                                                        SCIRun::StaticArray < CCVariable<double> >  &abskp,
                                                        SCIRun::StaticArray < CCVariable<double> >  &complexReal){

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter; 
    double VolFraction = VolFractionBC[c];
    double unscaled_weight;
    double unscaled_size;


    if ( VolFraction > 1.e-16 ){
      //now compute the particle values: 
      abskpt[c] = 0.0; 
      for ( int i = 0; i < Nqn; i++ ){ 

        unscaled_weight = (weights[i])[c];
        unscaled_size = (size[i])[c];

        if ( _p_planck_abskp ){ 

          abskp[i][c] = _part_radprops->planck_abs_coeff( unscaled_size, (pT[i])[c])* unscaled_weight;
          //double abskp_i = _part_radprops->planck_abs_coeff( unscaled_size, (pT[i])[c] );
          abskpt[c] += abskp[i][c] ; 
          
        } else if ( _p_ros_abskp ){ 

          abskp[i][c] = _part_radprops->ross_abs_coeff( unscaled_size, (pT[i])[c])* unscaled_weight;
          //double abskp_i = _part_rossprops->planck_abs_coeff( unscaled_size, (pT[i])[c] );
          abskpt[c] += abskp[i][c] ; 

        } 
      }


    }else{    

      abskpt[c] = 0.0;

    }
  }
}


void RadPropertyCalculator::constantCIF::compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                                         RadCalcSpeciesList size, RadCalcSpeciesList pT, RadCalcSpeciesList weights, 
                                                         const int Nqn, CCVariable<double>& scatkt,
                                                         SCIRun::StaticArray < CCVariable<double> > &scatktQuad,
                                                         SCIRun::StaticArray < CCVariable<double> >  &complexReal){

  for ( int i = 0; i < Nqn; i++ ){ 
    scatktQuad[i].allocate(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex());
    scatktQuad[i].initialize(0.0);
  }

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    IntVector c = *iter; 

    double VolFraction = VolFractionBC[c];
    double unscaled_weight;
    double unscaled_size;


    if ( VolFraction > 1.e-16 ){

      scatkt[c] = 0.0; 
      for ( int i = 0; i < Nqn; i++ ){ 

        unscaled_weight = (weights[i])[c];
        unscaled_size = (size[i])[c];
        if ( _p_planck_abskp ){ 
          double scatkt_i = _part_radprops->planck_sca_coeff( unscaled_size, (pT[i])[c]);
          scatktQuad[i][c]=scatkt_i* unscaled_weight;
          scatkt[c] += scatkt_i * unscaled_weight; 

        } else if ( _p_ros_abskp ){ 
          double scatkt_i =  _part_radprops->ross_sca_coeff( unscaled_size, (pT[i])[c] );
          scatktQuad[i][c]=scatkt_i* unscaled_weight;
          scatkt[c] += scatkt_i * unscaled_weight; 
        } 
      }
    }else{    

      scatkt[c] = 0.0;

    }
  }
}

void RadPropertyCalculator::constantCIF::computeComplexIndex( const Patch* patch,
                                                              constCCVariable<double>& VolFractionBC,
                                                              SCIRun::StaticArray < constCCVariable<double> > &composition,
                                                              SCIRun::StaticArray < CCVariable<double> >  &complexReal){
return;
}


void RadPropertyCalculator::constantCIF::computeAsymmetryFactor( const Patch* patch,
                                            constCCVariable<double>& VolFractionBC,
                            SCIRun::StaticArray < CCVariable<double> > &scatktQuad, 
                      SCIRun::StaticArray < constCCVariable<double> > &composition,
                                                       CCVariable<double>  &scatkt,
                                               CCVariable<double>  &asymmetryParam){

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter; 
    if ( VolFractionBC[c] > 1.e-16){
      for ( int i = 0; i <_nQn_part; i++ ){ 
      asymmetryParam[c] =_constAsymmFact ;
      }
    } // else, phase function remain zero in intrusions (set upstream).
  }
  return;
}


// This function does not need the particle temperature.
// Temperature is passed anyway so that other functions of 
// the base class can use it.
RadPropertyCalculator::basic::basic(const ProblemSpecP& db, bool scatteringOn){

  construction_success=true;
  _scatteringOn = scatteringOn; 

  if (scatteringOn){
    construction_success=false;
    proc0cout<<endl<<"Scattering not enabled for basic-radiative-particle-properties.  Use radprops, coal, OR turn off scattering!"<< "\n";;
  }

  db->findBlock( "particles" )->getWithDefault("Qabs",_Qabs,0.8); //  0.8 was used by Julien for coal particles

  if ( db->findBlock("particles")->findBlock("abskp") ){ 
    db->findBlock("particles")->findBlock("abskp")->getAttribute("label",_abskp_name); 
  }else{
    throw ProblemSetupException("Error: abskp name not found! This should be specified in the input file!",__FILE__, __LINE__);
  }

  //-----------------All class objects of this type should do this---------------------//
  db->findBlock("abskg")->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->require( "number_quad_nodes", _nQn_part ); 
  if ( _nQn_part ==0){
    construction_success = false;
  }
  _abskp_label = VarLabel::create(_abskp_name, CCVariable<double>::getTypeDescription() ); 
  _abskp_label_vector = vector<const VarLabel* >(_nQn_part);
  for (int i=0; i<_nQn_part; i++){
    std::stringstream out; 
    out << _abskp_name << "_" << i;
    _abskp_label_vector[i] = VarLabel::create(out.str(), CCVariable<double>::getTypeDescription() ); 
  }
  //-----------------------------------------------------------------------------------//
  _computeComplexIndex = false; // complex index of refraction not needed for this model
 
}

RadPropertyCalculator::basic::~basic(){
}


bool RadPropertyCalculator::basic::problemSetup(Task* tsk,int time_substep){
  return true;
}

// This is Julliens particle absorption coefficient model!
// In juliens model, Qabs, was hard coded to 0.8 for SUFCO coal.
void RadPropertyCalculator::basic::compute_abskp( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                                  RadCalcSpeciesList size, RadCalcSpeciesList pT, RadCalcSpeciesList weights, 
                                                  const int Nqn, CCVariable<double>& abskpt, 
                                                  SCIRun::StaticArray < CCVariable<double> >  &abskp,
                                                  SCIRun::StaticArray < CCVariable<double> >  &complexReal){
  double unscaled_weight;
  double unscaled_size;

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter; 
    if ( VolFractionBC[c] > 1.e-16 ){
      abskpt[c] =0.0 ; 
      for ( int i = 0; i < Nqn; i++ ){ 
        unscaled_weight = (weights[i])[c];
        unscaled_size = (size[i])[c];
        abskp[i][c]= M_PI/4.0*_Qabs*unscaled_weight*pow(unscaled_size,2.0);
        abskpt[c] +=abskp[i][c]; 
      }
    }else{    
      abskpt[c] = 0.0;
    }
  }
}


void RadPropertyCalculator::basic::compute_scatkt( const Patch* patch,  constCCVariable<double>& VolFractionBC,  
                                                   RadCalcSpeciesList size, RadCalcSpeciesList pT, RadCalcSpeciesList weights, 
                                                   const int Nqn, CCVariable<double>& scatkt,
                                                   SCIRun::StaticArray < CCVariable<double> > &scatktQuad,
                                                   SCIRun::StaticArray < CCVariable<double> >  &complexReal){

}

void RadPropertyCalculator::basic::computeComplexIndex( const Patch* patch,
                                                        constCCVariable<double>& VolFractionBC,
                                                        SCIRun::StaticArray < constCCVariable<double> > &composition,
                                                        SCIRun::StaticArray < CCVariable<double> >  &complexReal){
  return;
}


void RadPropertyCalculator::basic::computeAsymmetryFactor( const Patch* patch,
                                                           constCCVariable<double>& VolFractionBC,
                                                           SCIRun::StaticArray < CCVariable<double> > &scatktQuad, 
                                                           SCIRun::StaticArray < constCCVariable<double> > &composition,
                                                           CCVariable<double>  &scatkt,
                                                           CCVariable<double>  &asymmetryParam){
  return;
}
