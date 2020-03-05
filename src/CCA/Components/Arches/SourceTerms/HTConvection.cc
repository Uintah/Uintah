#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/HTConvection.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>


//===========================================================================

using namespace std;
using namespace Uintah; 

HTConvection::HTConvection( std::string src_name, ArchesLabel* field_labels,
    vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, field_labels->d_materialManager, req_label_names, type), _field_labels(field_labels)
{
  _source_grid_type = CC_SRC;

  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );

}

HTConvection::~HTConvection()
{
  VarLabel::destroy(ConWallHT_src_label);
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
HTConvection::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 
  const ProblemSpecP params_root = db->getRootNode();

  db->getWithDefault("dTCorrectionFactor", _dTCorrectionFactor,     2.0);
  //db->getWithDefault("temperature_label", _gas_temperature_name,    "temperature");
  //  db->getWithDefault("density_label",        _rho_name,                "density");

  std::string vol_frac= "volFraction";
  _volFraction_varlabel = VarLabel::find(vol_frac);
  //source terms name,and ...
  db->findBlock("ConWallHT_src")->getAttribute( "label", ConWallHT_src_name );
  _mult_srcs.push_back( ConWallHT_src_name );
  ConWallHT_src_label = VarLabel::create( ConWallHT_src_name, CCVariable<double>::getTypeDescription() );


  /*  if (params_root->findBlock("PhysicalConstants")) {
      ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
      db_phys->require("viscosity", _visc);
      if( _visc == 0 ) {
      throw InvalidValue("ERROR: HTConvection: problemSetup(): Zero viscosity specified in <PhysicalConstants> section of input file.",__FILE__,__LINE__);
      }
      } else {
      throw InvalidValue("ERROR: HTConvection: problemSetup(): Missing <PhysicalConstants> section in input file!",__FILE__,__LINE__);
      }
      */

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
HTConvection::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{

  std::string taskname = "HTConvection::eval";
  Task* tsk = scinew Task(taskname, this, &HTConvection::computeSource, timeSubStep);

  Ghost::GhostType  gac = Ghost::AroundCells;

  // get gas phase temperature label
  if (VarLabel::find("temperature")) {

    _gas_temperature_varlabel = VarLabel::find("temperature");

  } else {

    throw InvalidValue("ERROR: HT_convection: can't find gas phase temperature.",__FILE__,__LINE__);

  }

  Task::WhichDW which_dw;
  if (timeSubStep == 0) {

    tsk->computes(_src_label);
    tsk->computes(ConWallHT_src_label);
    which_dw = Task::OldDW;

  } else {

    which_dw = Task::NewDW;
    tsk->modifies(ConWallHT_src_label);
    tsk->modifies(_src_label);

  }

  tsk->requires( Task::OldDW, _volFraction_varlabel,      gac, 1 ); 
  tsk->requires( which_dw,     _gas_temperature_varlabel, gac, 1 );

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" )); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
HTConvection::computeSource( const ProcessorGroup* pc, 
    const PatchSubset*    patches, 
    const MaterialSubset* matls, 
    DataWarehouse*  old_dw, 
    DataWarehouse*  new_dw, 
    int             timeSubStep )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){



    Ghost::GhostType  gac = Ghost::AroundCells;
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 

    constCCVariable<double> rho;
    constCCVariable<double> gasT;
    constCCVariable<double> volFraction;

    CCVariable<double> rate; 
    CCVariable<double> ConWallHT_src;

    double rkg=0.0;
    double f_T_m=0.0; // face temperature on minus side
    double f_T_p=0.0; // face temperature on plus side
    double dT_dn=0.0;

    DataWarehouse* which_dw;
    if ( timeSubStep == 0 ){
      which_dw = old_dw;
      new_dw->allocateAndPut( rate,           _src_label,          matlIndex, patch );
      new_dw->allocateAndPut( ConWallHT_src,   ConWallHT_src_label, matlIndex, patch );
      rate.initialize(0.0);
      ConWallHT_src.initialize(0.0);

    } else {
      which_dw = new_dw;
      new_dw->getModifiable( rate, _src_label, matlIndex, patch );
      new_dw->getModifiable( ConWallHT_src,ConWallHT_src_label, matlIndex, patch );

    }

    old_dw->get( volFraction , _volFraction_varlabel , matlIndex , patch , gac , 1 );
    which_dw->get(gasT ,       _gas_temperature_varlabel, matlIndex , patch , gac, 1 );

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    IntVector lowPindex = patch->getCellLowIndex();
    IntVector highPindex = patch->getCellHighIndex();
    //Pad for ghosts
    lowPindex -= IntVector(1,1,1);
    highPindex += IntVector(1,1,1);

    Vector Dx = patch->dCell(); 

    Uintah::parallel_for(range, [&](int i, int j, int k) {

        //bool exper=0;
        //exper=patch->containsIndex(lowPindex, highPindex, IntVector(i,j,k) ); 

        double cellvol=Dx.x()*Dx.y()*Dx.z();
        double delta_n = 0.0; 
        double total_area_face=0.0;

        rate(i,j,k)=0.0; 
        ConWallHT_src(i,j,k)=0.0; 

        // for computing dT/dn we have 2 scenarios:
        // (1) cp is a wall:
        //         gasT[cp]          gasT[c] 
        // dT_dn = _________   _      ________________
        //          delta_n            delta_n
        // (2) cm is a wall:
        //         gasT[c]              gasT[cm]
        // Note: Surface temperature of the wall is stored at the cell center.


        if ( volFraction(i,j,k) > 0. ){ 
          // fluid cell  
          if (patch->containsIndex(lowPindex, highPindex, IntVector(i+1,j,k) )){ 
            delta_n=Dx.x();
            f_T_p=gasT(i+1,j,k);
            f_T_m=gasT(i,j,k);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            rate(i,j,k)          = volFraction(i+1,j,k) < 1.0 ?  rate(i,j,k)+rkg*dT_dn*_dTCorrectionFactor/delta_n :rate(i,j,k);// w/m^3
          }


          if (patch->containsIndex(lowPindex, highPindex, IntVector(i-1,j,k) )) {
            delta_n=Dx.x();
            f_T_p=gasT(i,j,k);
            f_T_m=gasT(i-1,j,k);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            rate(i,j,k) = volFraction(i-1,j,k) < 1.0 ?  rate(i,j,k)-rkg*dT_dn*_dTCorrectionFactor/delta_n :rate(i,j,k);// w/m^3
          }


          // Next for the Y direction 
          if (patch->containsIndex(lowPindex, highPindex, IntVector(i,j+1,k) )){ 
            delta_n=Dx.y();
            f_T_p=gasT(i,j+1,k);
            f_T_m=gasT(i,j,k);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            rate(i,j,k) = volFraction(i,j+1,k) < 1.0 ?  rate(i,j,k)+rkg*dT_dn*_dTCorrectionFactor/delta_n :rate(i,j,k);// w/m^3
          }

          if (patch->containsIndex(lowPindex, highPindex, IntVector(i,j-1,k) )){ 
            delta_n=Dx.y();
            f_T_p=gasT(i,j,k);
            f_T_m=gasT(i,j-1,k);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            rate(i,j,k) = volFraction(i,j-1,k) < 1.0 ?  rate(i,j,k)-rkg*dT_dn*_dTCorrectionFactor/delta_n :rate(i,j,k);// w/m^3
          }

          // Next for the z direction 
          if (patch->containsIndex(lowPindex, highPindex, IntVector(i,j,k+1) )){ 
            delta_n=Dx.z();
            f_T_p=gasT(i,j,k+1);
            f_T_m=gasT(i,j,k);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            rate(i,j,k) = volFraction(i,j,k+1) < 1.0 ?  rate(i,j,k)+rkg*dT_dn*_dTCorrectionFactor/delta_n :rate(i,j,k);// w/m^3
          }


          if (patch->containsIndex(lowPindex, highPindex, IntVector(i,j,k-1) )) {
            delta_n=Dx.z();
            f_T_p=gasT(i,j,k);
            f_T_m=gasT(i,j,k-1);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            rate(i,j,k) = volFraction(i,j,k-1) < 1.0 ?  rate(i,j,k)-rkg*dT_dn*_dTCorrectionFactor/delta_n :rate(i,j,k);// w/m^3
          }      



        }// end for fluid cell

        if(volFraction(i,j,k) < 1.0){

          // wall cells(i,j,k)
          // x+ direction
          //  std::cout<<i<<j<<k<<"\n";
          if (patch->containsIndex(lowPindex, highPindex, IntVector(i+1,j,k) )){ 
            delta_n=Dx.x();
            f_T_p=gasT(i+1,j,k);
            f_T_m=gasT(i,j,k);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            ConWallHT_src(i,j,k) = volFraction(i+1,j,k) > 0.0 ?  ConWallHT_src(i,j,k)+rkg*dT_dn*_dTCorrectionFactor/delta_n :ConWallHT_src(i,j,k);// w/m^3
            total_area_face =      volFraction(i+1,j,k) > 0.0 ?  total_area_face+Dx.y()*Dx.z() :total_area_face;// w/m^3
          }


          // x- direction 
          if (patch->containsIndex(lowPindex, highPindex, IntVector(i-1,j,k) )) {
            delta_n=Dx.x();
            f_T_p=gasT(i,j,k);
            f_T_m=gasT(i-1,j,k);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            ConWallHT_src(i,j,k) = volFraction(i-1,j,k) > 0.0 ?  ConWallHT_src(i,j,k)-rkg*dT_dn*_dTCorrectionFactor/delta_n :ConWallHT_src(i,j,k);// w/m^3
            total_area_face =      volFraction(i-1,j,k) > 0.0 ?  total_area_face+Dx.y()*Dx.z() :total_area_face;// w/m^3
          }


          // Next for the Y+ direction 
          if (patch->containsIndex(lowPindex, highPindex, IntVector(i,j+1,k) )){ 
            delta_n=Dx.y();
            f_T_p=gasT(i,j+1,k);
            f_T_m=gasT(i,j,k);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            ConWallHT_src(i,j,k) = volFraction(i,j+1,k) > 0.0 ?  ConWallHT_src(i,j,k)+rkg*dT_dn*_dTCorrectionFactor/delta_n :ConWallHT_src(i,j,k);// w/m^3
            total_area_face =      volFraction(i,j+1,k) > 0.0 ?  total_area_face+Dx.x()*Dx.z() :total_area_face;// w/m^3
          } 


          // Next for the Y- direction 
          if (patch->containsIndex(lowPindex, highPindex, IntVector(i,j-1,k) )){ 
            delta_n=Dx.y();
            f_T_p=gasT(i,j,k);
            f_T_m=gasT(i,j-1,k);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            ConWallHT_src(i,j,k) = volFraction(i,j-1,k) > 0.0 ?  ConWallHT_src(i,j,k)-rkg*dT_dn*_dTCorrectionFactor/delta_n :ConWallHT_src(i,j,k);// w/m^3
            total_area_face =      volFraction(i,j-1,k) > 0.0 ?  total_area_face+Dx.x()*Dx.z() :total_area_face;// w/m^3
          }

          // Next for the z direction 
          if (patch->containsIndex(lowPindex, highPindex, IntVector(i,j,k+1) )){ 
            delta_n=Dx.z();
            f_T_p=gasT(i,j,k+1);
            f_T_m=gasT(i,j,k);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            ConWallHT_src(i,j,k) = volFraction(i,j,k+1) > 0.0 ?  ConWallHT_src(i,j,k)+rkg*dT_dn*_dTCorrectionFactor/delta_n :ConWallHT_src(i,j,k);// w/m^3
            total_area_face =      volFraction(i,j,k+1) > 0.0 ?  total_area_face+Dx.x()*Dx.y() :total_area_face;// w/m^3
          }


          if (patch->containsIndex(lowPindex, highPindex, IntVector(i,j,k-1) )) {
            delta_n=Dx.z();
            f_T_p=gasT(i,j,k);
            f_T_m=gasT(i,j,k-1);
            dT_dn = (f_T_p - f_T_m) / delta_n;
            rkg = ThermalConductGas(f_T_p, f_T_m); // [=] J/s/m/K
            ConWallHT_src(i,j,k) = volFraction(i,j,k-1) > 0.0 ?  ConWallHT_src(i,j,k)-rkg*dT_dn*_dTCorrectionFactor/delta_n :ConWallHT_src(i,j,k);// w/m^3
            total_area_face =      volFraction(i,j,k-1) > 0.0 ?  total_area_face+Dx.x()*Dx.y() :total_area_face;// w/m^3
          }
          ConWallHT_src(i,j,k) =total_area_face>0.0 ?  ConWallHT_src(i,j,k)/total_area_face*cellvol :0.0;//W/m2

        }

    }); // end fluid cell loop for computing the 

  }// end patch loop
}
//---------------------------------------------------------------------------
// Method: Wall heat losss by conduction terms for the gas phase enthalpy  Source Terms
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
HTConvection::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "HTConvection::initialize"; 

  Task* tsk = scinew Task(taskname, this, &HTConvection::initialize);

  tsk->computes(_src_label);
  tsk->computes(ConWallHT_src_label);

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

}
void 
HTConvection::initialize( const ProcessorGroup* pc, 
    const PatchSubset* patches, 
    const MaterialSubset* matls, 
    DataWarehouse* old_dw, 
    DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 

    CCVariable<double> src;
    CCVariable<double> ConWallHT_src;
    new_dw->allocateAndPut( src,          _src_label, matlIndex, patch ); 
    new_dw->allocateAndPut( ConWallHT_src, ConWallHT_src_label, matlIndex, patch ); 

    src.initialize(0.0); 
    ConWallHT_src.initialize(0.0);
  }
}


double
HTConvection::ThermalConductGas(double Tg, double Tp){

  double tg0[10] = {300.,  400.,   500.,   600.,  700.,  800.,  900.,  1000., 1100., 1200. };
  double kg0[10] = {.0262, .03335, .03984, .0458, .0512, .0561, .0607, .0648, .0685, .07184};
  double T = (Tp+Tg)/2; // Film temperature

  //   CALCULATE UG AND KG FROM INTERPOLATION OF TABLE VALUES FROM HOLMAN
  //   FIND INTERVAL WHERE TEMPERATURE LIES.

  double kg = 0.0;

  if( T > 1200.0 ) {
    kg = kg0[9] * pow( T/tg0[9], 0.58);

  } else if ( T < 300 ) {
    kg = kg0[0];

  } else {
    int J = -1;
    for ( int I=0; I < 9; I++ ) {
      if ( T > tg0[I] ) {
        J = J + 1;
      }
    }
    double FAC = ( tg0[J] - T ) / ( tg0[J] - tg0[J+1] );
    kg = ( -FAC*( kg0[J] - kg0[J+1] ) + kg0[J] );
  }

  return kg; // I believe this is in J/s/m/K, but not sure
}
