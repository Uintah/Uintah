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
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
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

  for ( ProblemSpecP db_pc = db->findBlock("calculator"); db_pc != nullptr; db_pc = db_pc->findNextBlock("calculator") ) {

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

  //if ( test_label == 0 ){ 
    //_abskg_label = VarLabel::create(_abskg_name, CCVariable<double>::getTypeDescription() ); 
  //} else { 
    //throw ProblemSetupException("Error: Abskg label already used for constant properties: "+_abskg_name,__FILE__, __LINE__);
  //}

  bool property_on = true; 

  return property_on; 
}

void RadPropertyCalculator::ConstantProperties::initialize_abskg( const Patch* patch, CCVariable<double>& abskg ){ 
  abskg.initialize(_abskg_value); 
}

    
void RadPropertyCalculator::ConstantProperties::compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, 
                                                              RadCalcSpeciesList species, constCCVariable<double>& mixT, 
                                                              CCVariable<double>& abskg ){ 
  abskg.initialize(_abskg_value); 
}

RadPropertyCalculator::specialProperties::specialProperties() {

};

RadPropertyCalculator::specialProperties::~specialProperties() {};

bool RadPropertyCalculator::specialProperties::problemSetup( const ProblemSpecP& db ) {
        
  ProblemSpecP db_prop = db; 
  db_prop->getWithDefault("expressionNumber",_expressionNumber,1); 

  if ( db_prop->findBlock("abskg") ){ 
    db_prop->findBlock("abskg")->getAttribute("label",_abskg_name); 
  } else { 
    _abskg_name = "abskg"; 
  }


  //Create the particle absorption coeff as a <PropertyModel>

  //if ( test_label == 0 ){ 
    //_abskg_label = VarLabel::create(_abskg_name, CCVariable<double>::getTypeDescription() ); 
  //} else { 
    //throw ProblemSetupException("Error: Abskg label already used for constant properties: "+_abskg_name,__FILE__, __LINE__);
  //}

  bool property_on = true; 

  return property_on; 
}

void RadPropertyCalculator::specialProperties::initialize_abskg( const Patch* patch, CCVariable<double>& abskg ){ 
  abskg.initialize(0.0); 
}
void RadPropertyCalculator::specialProperties::compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, 
                                                              RadCalcSpeciesList species, constCCVariable<double>& mixT, 
                                                              CCVariable<double>& abskg ){ 
  if (_expressionNumber == 1){
    for (CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter){ 
      IntVector c = *iter; 
      abskg[c]=5.0*exp((pow(mixT[c]/1000.0,4.0)-1.0)/2.0*(-2.0)-0.1);
    }
  }
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

  //if ( test_label == 0 ){ 
    //_abskg_label = VarLabel::create(_abskg_name, CCVariable<double>::getTypeDescription() ); 
  //} else { 
    //throw ProblemSetupException("Error: Abskg label already used for constant properties: "+_abskg_name,__FILE__, __LINE__);
  //}

  bool property_on = true; 
  return property_on; 
}

void RadPropertyCalculator::BurnsChriston::initialize_abskg( const Patch* patch, CCVariable<double>& abskg ){ 
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
void
RadPropertyCalculator::BurnsChriston::compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, constCCVariable<double>& mixT, CCVariable<double>& abskg ){ 
  
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
//  Weighted Sum of Grey Gases from G. Krishnamoorthy 2013 
// ---------------------------------------
RadPropertyCalculator::GauthamWSGG::GauthamWSGG(){

// Set up coefficients from Krishnamoorthy 2013
 _K =vector< vector<double> > (4,vector<double> (4,0.0));
 _C1=vector< vector<double> > (4,vector<double> (4,0.0));
 _C2=vector< vector<double> > (4,vector<double> (4,0.0));

_K[0][0]=0.06592; _K[0][1]=0.99698; _K[0][2]=10.00038; _K[0][3]=100.00000;
_K[1][0]=0.10411; _K[1][1]=1.00018; _K[1][2]=9.99994;  _K[1][3]=100.00000;
_K[2][0]=0.20616; _K[2][1]=1.39587; _K[2][2]=8.56904;  _K[2][3]=99.75698;
_K[3][0]=0.21051; _K[3][1]=1.33782; _K[3][2]=8.55495;  _K[3][3]=99.75649;

_C1[0][0]=7.85445e-5; _C1[0][1]=-9.47416e-5; _C1[0][2]=-5.51091e-5; _C1[0][3]=7.26634e-6;
_C1[1][0]=9.33340e-5; _C1[1][1]=-5.32833e-5; _C1[1][2]=-1.01806e-4; _C1[1][3]=-2.25973e-5;
_C1[2][0]=9.22363e-5; _C1[2][1]=-4.25444e-5; _C1[2][2]=-9.89282e-5; _C1[2][3]=-3.83770e-5;
_C1[3][0]=1.07579e-4; _C1[3][1]=-3.09769e-5; _C1[3][2]=-1.13634e-4; _C1[3][3]=-3.43141e-5;

_C2[0][0]=2.39641e-1; _C2[0][1]=3.42342e-1; _C2[0][2]=1.37773e-1; _C2[0][3]=4.40724e-2;
_C2[1][0]=1.89029e-1; _C2[1][1]=2.87021e-1; _C2[1][2]=2.54516e-1; _C2[1][3]=6.54289e-2;
_C2[2][0]=1.91464e-1; _C2[2][1]=2.34876e-1; _C2[2][2]=2.47320e-1; _C2[2][3]=9.59426e-2;
_C2[3][0]=1.54129e-1; _C2[3][1]=2.43637e-1; _C2[3][2]=2.84084e-1; _C2[3][3]=8.57853e-2;


};


RadPropertyCalculator::GauthamWSGG::~GauthamWSGG() {};
    
bool 
RadPropertyCalculator::GauthamWSGG::problemSetup( const ProblemSpecP& db ) {
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

  //if ( test_label == 0 ){ 
    //_abskg_label = VarLabel::create(_abskg_name, CCVariable<double>::getTypeDescription() ); 
  //} else { 
    //throw ProblemSetupException("Error: Abskg label already created before GauthamWSGG Radproperties: "+_abskg_name,__FILE__, __LINE__);
  //}

    db_h->getWithDefault("mix_mol_w_label",_mixMolecWeight,"mixture_molecular_weight"); 

    _sp_mw = vector < double> (0);
    _sp_mw.push_back(1.0/44.0); // CO2
    _sp_mw.push_back(1.0/18.0); // H2O




  bool property_on = true;
  return property_on; 

}

void RadPropertyCalculator::GauthamWSGG::initialize_abskg( const Patch* patch, CCVariable<double>& abskg ){ 
  abskg.initialize(0.0); 
}

void 
RadPropertyCalculator::GauthamWSGG::compute_abskg( const Patch* patch, 
    constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, 
    constCCVariable<double>& mixT, CCVariable<double>& abskg ){ 


  vector< double > a(4,0.0);
  int ii;
  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    if (VolFractionBC[c] > 0.0){



      // compute molfraction from mass fraction
      double molFracCO2 =   (species[0])[c] * _sp_mw[0] * 1.0 / (species[2])[c]  ;
      double molFracH2O =   (species[1])[c] * _sp_mw[1] * 1.0 / (species[2])[c]  ;
      double totalAttenGas= molFracCO2+molFracH2O;  // total attenuating gases
      double Ratio =  molFracCO2==0.0 ? 1e6 : molFracH2O/molFracCO2;


      if (Ratio <= 0.2){
        ii =0;
      } else if ( Ratio  <= 0.67){
        ii =1;
      } else if ( Ratio  <= 1.50){
        ii =2;
      } else if ( Ratio  <= 4.00){
        ii =3;
      } else {
        ii =3; // throw error? Extrapolate?
      }

      double emissivity =0.0; 

      for (int i=0; i<4; i++){
          a[i]    =  _C1[ii][i]  *   mixT[c]      + _C2[ii][i];
      // ^weight^    ^C1^         ^Temperature(K)^  ^C2^
      }

      for (int i=0; i<4; i++){
        emissivity += a[i]*(1.0 - exp( -_K[ii][i] * (totalAttenGas) *  d_opl));
      }
      abskg[c] = 1.0/ d_opl * -log(1.0-emissivity)*d_gasPressure;

    } else{
      abskg[c] = 1.0; // emissivity of wall = 1;
    }

  }

}



vector<std::string> 
RadPropertyCalculator::GauthamWSGG::get_sp(){

  vector<std::string> temp (0);
  temp.push_back(_co2_name);
  temp.push_back(_h2o_name);
  temp.push_back(_mixMolecWeight);
   return temp;

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

  //if ( test_label == 0 ){ 
    //_abskg_label = VarLabel::create(_abskg_name, CCVariable<double>::getTypeDescription() ); 
  //} else { 
    //throw ProblemSetupException("Error: Abskg label already used for constant properties: "+_abskg_name,__FILE__, __LINE__);
  //}



  bool property_on = true;
  return property_on; 

}

void RadPropertyCalculator::HottelSarofim::initialize_abskg( const Patch* patch, CCVariable<double>& abskg ){ 
  abskg.initialize(0.0); 
}

void 
RadPropertyCalculator::HottelSarofim::compute_abskg( const Patch* patch, 
    constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species, 
    constCCVariable<double>& mixT, CCVariable<double>& abskg ){ 

  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

  fort_hottel(idxLo, idxHi, mixT,
              species[0], species[1], VolFractionBC,
              d_opl, species[2], abskg,d_gasPressure);

}



vector<std::string> 
RadPropertyCalculator::HottelSarofim::get_sp(){

  vector<std::string> temp (0);
  temp.push_back(_co2_name);
  temp.push_back(_h2o_name);
  temp.push_back(_soot_name);
   return temp;

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
    
  if ( _gg_radprops != nullptr ) 
    delete _gg_radprops; 
}
    
bool RadPropertyCalculator::RadPropsInterface::problemSetup( const ProblemSpecP& db ) {

  if ( db->findBlock( "grey_gas" ) ){

    ProblemSpecP db_gg = db->findBlock( "grey_gas" );

    db_gg->getWithDefault("mix_mol_w_label",_mix_mol_weight_name,"mixture_molecular_weight"); 
    std::string inputfile;
    db_gg->require("inputfile",inputfile); 

    //allocate gray gas object: 
    _gg_radprops = scinew RadProps::GreyGas( inputfile ); 

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
  }
  else { 
    _abskg_name = "abskg"; 
  }

  //if ( test_label == 0 ){ 
    //_abskg_label = VarLabel::create(_abskg_name, CCVariable<double>::getTypeDescription() ); 
  //} else { 
    //throw ProblemSetupException("Error: Abskg label already in use: "+_abskg_name,__FILE__, __LINE__);
  //}

  return true; 
}
    
void RadPropertyCalculator::RadPropsInterface::initialize_abskg( const Patch* patch, CCVariable<double>& abskg ){ 
  abskg.initialize(0.0); 
}

void
RadPropertyCalculator::RadPropsInterface::compute_abskg( const Patch* patch, constCCVariable<double>& VolFractionBC, RadCalcSpeciesList species,  constCCVariable<double>& mixT, CCVariable<double>& abskg)
{ 

  int N = species.size(); 
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

      _gg_radprops->mixture_coeffs(effCff, mol_frac, mixT[c], RadProps::EFF_ABS_COEFF);

      abskg[c] = effCff*100.0*d_gasPressure; // from cm^-1 to m^-1 //need to generalize this to the other coefficients


    } else { 

   //   abskg[c] = 1.0; 

    }
  }
}




#endif 


