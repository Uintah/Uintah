#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <CCA/Components/Arches/SourceTerms/MonoSoot.h>
//===========================================================================

using namespace std;
using namespace Uintah;

MonoSoot::MonoSoot( std::string src_name, ArchesLabel* field_labels,
                                                    vector<std::string> req_label_names, std::string type )
: SourceTermBase(src_name, field_labels->d_materialManager, req_label_names, type), _field_labels(field_labels)
{

  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
  _source_grid_type = CC_SRC;
}

MonoSoot::~MonoSoot()
{
      VarLabel::destroy(m_tar_src_label      );
      VarLabel::destroy(m_nd_src_label       );
      VarLabel::destroy(m_soot_mass_src_label);
      VarLabel::destroy(m_balance_src_label  );
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
MonoSoot::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb;

  db->getWithDefault("mix_mol_weight_label", m_mix_mol_weight_name,   "mixture_molecular_weight");
  db->getWithDefault("tar_label",            m_tar_name,              "Tar");
  db->getWithDefault("Ysoot_label",          m_Ysoot_name,            "Ysoot");
  db->getWithDefault("Ns_label",             m_Ns_name,               "Ns");
  db->getWithDefault("o2_label",             m_O2_name,               "O2");
  db->getWithDefault("oh_label",             m_OH_name,               "OH");
  db->getWithDefault("co2_label",	           m_CO2_name,		          "CO2");
  db->getWithDefault("h2o_label",            m_H2O_name,              "H2O");
  db->getWithDefault("h2o_label",            m_H_name,                "H");
  db->getWithDefault("h2o_label",            m_H2_name,               "H2");
  db->getWithDefault("h2o_label",            m_C2H2_name,             "C2H2");
  db->getWithDefault("density_label",        m_rho_name,              "density");
  db->getWithDefault("temperature_label",    m_temperature_name,      "radiation_temperature");

  db->findBlock("tar_src")->getAttribute( "label", m_tar_src_name );
  db->findBlock("num_density_src")->getAttribute( "label", m_nd_name );
  db->findBlock("soot_mass_src")->getAttribute( "label", m_soot_mass_name );
  db->findBlock("mass_balance_src")->getAttribute( "label", m_balance_name );

  // Since we are producing multiple sources, we load each name into this vector
  // so that we can do error checking upon src term retrieval.
  _mult_srcs.push_back( m_tar_src_name );
  _mult_srcs.push_back( m_nd_name );
  _mult_srcs.push_back( m_soot_mass_name );
  _mult_srcs.push_back( m_balance_name );

  m_tar_src_label       = VarLabel::create( m_tar_src_name, CCVariable<double>::getTypeDescription() );
  m_nd_src_label        = VarLabel::create( m_nd_name, CCVariable<double>::getTypeDescription() );
  m_soot_mass_src_label = VarLabel::create( m_soot_mass_name, CCVariable<double>::getTypeDescription() );
  m_balance_src_label   = VarLabel::create( m_balance_name, CCVariable<double>::getTypeDescription() );

  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species( m_O2_name );
  helper.add_lookup_species( m_OH_name );
  helper.add_lookup_species( m_rho_name );
  helper.add_lookup_species( m_CO2_name );
  helper.add_lookup_species( m_H2O_name );
  helper.add_lookup_species( m_H_name );
  helper.add_lookup_species( m_H2_name );
  helper.add_lookup_species( m_C2H2_name );
  helper.add_lookup_species( m_mix_mol_weight_name );
  //_field_labels->add_species( m_temperature_name );
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
MonoSoot::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{


  std::string taskname = "MonoSoot::eval";
  Task* tsk = scinew Task(taskname, this, &MonoSoot::computeSource, timeSubStep);

  Task::WhichDW which_dw;
  if (timeSubStep == 0) {
    tsk->computes(m_tar_src_label);
    tsk->computes(m_nd_src_label);
    tsk->computes(m_soot_mass_src_label);
    tsk->computes(m_balance_src_label);
    which_dw = Task::OldDW;
  } else {
    which_dw = Task::NewDW;
    tsk->modifies(m_tar_src_label);
    tsk->modifies(m_nd_src_label);
    tsk->modifies(m_soot_mass_src_label);
    tsk->modifies(m_balance_src_label);
  }
  // resolve some labels:
  m_mix_mol_weight_label  = VarLabel::find( m_mix_mol_weight_name);
  m_tar_label             = VarLabel::find( m_tar_name);
  m_Ysoot_label           = VarLabel::find( m_Ysoot_name);
  m_Ns_label              = VarLabel::find( m_Ns_name);
  m_o2_label              = VarLabel::find( m_O2_name);
  m_oh_label              = VarLabel::find( m_OH_name);
  m_co2_label             = VarLabel::find( m_CO2_name);
  m_h2o_label             = VarLabel::find( m_H2O_name);
  m_h_label               = VarLabel::find( m_H_name);
  m_h2_label              = VarLabel::find( m_H2_name);
  m_c2h2_label            = VarLabel::find( m_C2H2_name);
  m_temperature_label     = VarLabel::find( m_temperature_name);
  m_rho_label             = VarLabel::find( m_rho_name);

  tsk->requires( which_dw, m_mix_mol_weight_label,               Ghost::None, 0 );
  tsk->requires( which_dw, m_tar_label,                          Ghost::None, 0 );
  tsk->requires( which_dw, m_Ysoot_label,                        Ghost::None, 0 );
  tsk->requires( which_dw, m_Ns_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_o2_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_oh_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_co2_label,                          Ghost::None, 0 );
  tsk->requires( which_dw, m_h2o_label,                          Ghost::None, 0 );
  tsk->requires( which_dw, m_h_label,                            Ghost::None, 0 );
  tsk->requires( which_dw, m_h2_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_c2h2_label,                         Ghost::None, 0 );
  tsk->requires( which_dw, m_temperature_label,                  Ghost::None, 0 );
  tsk->requires( which_dw, m_rho_label,                          Ghost::None, 0 );

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));
  
  //get the system pressure:
  ChemHelper& helper = ChemHelper::self();
  ChemHelper::TableConstantsMapType tab_constants = helper.get_table_constants();
  auto i_press = tab_constants->find("Pressure");
  if ( i_press != tab_constants->end() ){
    m_sys_pressure = i_press->second;
  } else {
    m_sys_pressure = 101325.0; //otherise assume atmospheric
  }

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
MonoSoot::computeSource( const ProcessorGroup* pc,
                          const PatchSubset*    patches,
                          const MaterialSubset* matls,
                          DataWarehouse*  old_dw,
                          DataWarehouse*  new_dw,
                          int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> tar_src;
    CCVariable<double> num_density_src;
    CCVariable<double> soot_mass_src;
    CCVariable<double> balance_src;

    constCCVariable<double> mix_mol_weight;
    constCCVariable<double> Tar;
    constCCVariable<double> Ysoot;
    constCCVariable<double> Ns;
    constCCVariable<double> O2;
    constCCVariable<double> OH;
    constCCVariable<double> CO2;
    constCCVariable<double> H2O;
    constCCVariable<double> H2;
    constCCVariable<double> H;
    constCCVariable<double> C2H2;
    constCCVariable<double> rho;
    constCCVariable<double> temperature;

    DataWarehouse* which_dw;
    if ( timeSubStep == 0 ){
        which_dw = old_dw;
        new_dw->allocateAndPut( tar_src, m_tar_src_label, matlIndex, patch );
        new_dw->allocateAndPut( num_density_src, m_nd_src_label, matlIndex, patch );
        new_dw->allocateAndPut( soot_mass_src, m_soot_mass_src_label, matlIndex, patch );
        new_dw->allocateAndPut( balance_src, m_balance_src_label, matlIndex, patch );
        tar_src.initialize(0.0);
        num_density_src.initialize(0.0);
        soot_mass_src.initialize(0.0);
        balance_src.initialize(0.0);
    } else {
        which_dw = new_dw;
        new_dw->getModifiable( tar_src, m_tar_src_label, matlIndex, patch );
        new_dw->getModifiable( num_density_src, m_nd_src_label, matlIndex, patch );
        new_dw->getModifiable( soot_mass_src, m_soot_mass_src_label, matlIndex, patch );
        new_dw->getModifiable( balance_src, m_balance_src_label, matlIndex, patch );
    }

    which_dw->get( mix_mol_weight , m_mix_mol_weight_label , matlIndex , patch , gn, 0 );
    which_dw->get( Tar            , m_tar_label            , matlIndex , patch , gn, 0 );
    which_dw->get( Ysoot          , m_Ysoot_label          , matlIndex , patch , gn, 0 );
    which_dw->get( Ns             , m_Ns_label             , matlIndex , patch , gn, 0 );
    which_dw->get( O2             , m_o2_label             , matlIndex , patch , gn, 0 );
    which_dw->get( OH             , m_oh_label             , matlIndex , patch , gn, 0 );
    which_dw->get( CO2            , m_co2_label            , matlIndex , patch , gn, 0 );
    which_dw->get( H2O            , m_h2o_label            , matlIndex , patch , gn, 0 );
    which_dw->get( H              , m_h_label              , matlIndex , patch , gn, 0 );
    which_dw->get( H2             , m_h2_label             , matlIndex , patch , gn, 0 );
    which_dw->get( C2H2           , m_c2h2_label           , matlIndex , patch , gn, 0 );
    which_dw->get( temperature    , m_temperature_label    , matlIndex , patch , gn, 0 );
    which_dw->get( rho            , m_rho_label            , matlIndex , patch , gn, 0 );

    /// Obtain time-step length
    delt_vartype DT;
    old_dw->get( DT,_field_labels->d_delTLabel);
    const double delta_t = DT;

    const double rhos = 1850.;           ///< soot density                   (kg/m3)
    const double kB   = 1.3806485279e-23; ///< Boltzmann constant             (kg m2/s2 K)
    const double Na   = 6.0221413e26;     ///< Avogadro's number              (#/kmole)
    const double Rgas = kB*Na;            ///< Ideal gas constant             (J/K kmole)
    const double Visc = 5.5e-5;           ///< Gas viscosity approximation    (kg/m s)
    const double MWc  = 12.011/Na;        ///< molecular weight c             (kg)
    const double mtar = 350.0/Na;         ///< molecular weight tar           (kg)

    const double da   = 1.395e-10*pow(3.0,0.5); ///< Diameter of a single aromatic ring (m)
    const double eps  = 2.2;              ///< Van der Waals enhancement factor
    const double chiC  = 2.3e19;         ///< sites/m2
    const double mC2H2 = 26.0/Na;        ///< kg/site

    const double AO2 = 7.98E-1;     ///< pre-exponential constant:  kg*K^0.5/Pa*m2*s
    const double EO2 = -1.77E8;     ///< activation energy:         J/kmole
    const double AOH = 1.89E-3;     ///< pre-exponentail constant:  kg*K^0.5/Pa*m2*s
    const double ACO2 = 3.06E-17;   ///< pre-exponential constant:  kg/Pa^0.5*K2*m2*s
    const double ECO2 = -5.56E6;    ///< activation energy:         J/kmole
    const double AH2O = 6.27E4;     ///< pre-exponentail constant:  kg*K^0.5/Pa^1.21*m2*s
    const double EH2O = -2.95E8;    ///< activation energy:         J/kmole

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        
        IntVector c = *iter;
        const double XO2    = ( mix_mol_weight[c] > 1.0e-10 ) ?
                              O2[c] * 1.0 / (mix_mol_weight[c] * 32.0)   : 0.0;
        const double XOH    = ( mix_mol_weight[c] > 1.0e-10 ) ?
                              OH[c] * 1.0 / (mix_mol_weight[c] * 17.01)  : 0.0;
        const double XCO2   = ( mix_mol_weight[c] > 1.0e-10 ) ?
                              CO2[c] * 1.0 / (mix_mol_weight[c] * 44.0)  : 0.0;
        const double XH2O   = ( mix_mol_weight[c] > 1.0e-10 ) ?
                              H2O[c] * 1.0 / (mix_mol_weight[c] * 18.02) : 0.0;
        const double XC2H2  = ( mix_mol_weight[c] > 1.0e-10 ) ?
                              C2H2[c] * 1.0 / (mix_mol_weight[c] * 26.0) : 0.0;
        const double XH2    = ( mix_mol_weight[c] > 1.0e-10 ) ?
                              H2[c] * 1.0 / (mix_mol_weight[c] * 2.0) : 0.0;
        const double XH     = ( mix_mol_weight[c] > 1.0e-10 ) ?
                              H[c] * 1.0 / (mix_mol_weight[c] * 1.0) : 0.0;
        const double Nsoot  = (Ns[c] > 0.0 and Ysoot[c] > 0.0) ? Ns[c] * rho[c]    : 0.0;
        const double Msoot  = (Ns[c] > 0.0 and Ysoot[c] > 0.0) ? Ysoot[c] * rho[c] : 0.0;
        const double Ntar   = (Tar[c] > 0.0)   ? Tar[c]   * rho[c] / mtar : 0.0;
        
        const double T = abs(temperature[c]);
        const double P = m_sys_pressure;
        const double dt = delta_t;
            
        //----- Surface Consumption
        const double Oxidation    = (AO2*P*XO2*exp(EO2/Rgas/abs(T))+AOH*P*XOH)/pow(abs(T),0.5);
        const double Gasification =  ACO2*pow(P*XCO2,0.5)*pow(abs(T),2.0)*exp(ECO2/Rgas/abs(T))+AH2O*pow(abs(T),-0.5)*pow(P*XH2O,0.13)*exp(EH2O/Rgas/abs(T));
        
        //----- HACA constants
        const double k1  = 4.2e10*exp(-5.4392e7/Rgas/abs(T));                  ///< m3/kmole*s
        const double kr1 = 3.9e9*exp(-4.6024e7/Rgas/abs(T));                  ///< m3/kmole*s
        const double k2  = 1.0e7*pow(abs(T),0.734)*exp(-5.98312e6/Rgas/abs(T)); ///< m3/kmole*s
        const double kr2 = 3.68e5*pow(abs(T),1.139)*exp(-7.15464e6/Rgas/abs(T));///< m3/kmole*s
        const double k3  = 2.0e10;                                             ///< m3/kmole*s
        const double k4  = 8.0e4*exp(-1.58992e7/Rgas/abs(T));                 ///< m3/kmole*s

        const double CC2H2 = P*XC2H2/Rgas/abs(T);   ///< kmole/m3
        const double CH    = P*XH/Rgas/abs(T);      ///< kmole/m3
        const double CH2   = P*XH2/Rgas/abs(T);     ///< kmole/m3
        const double COH   = P*XOH/Rgas/abs(T);     ///< kmole/m3
        const double CH2O  = P*XH2O/Rgas/abs(T);    ///< kmole/m3

        const double a = 12.65-0.00563*abs(T);
        const double b = -1.38+0.00068*abs(T);
            
        const double TarAlpha  = abs(tanh(a/log10(mtar)+b));
        const double HACAp = k4*CC2H2*TarAlpha*chiC*(k1*CH+k2*COH)*mC2H2/(kr1*CH2+kr2*CH2O+k3*CH+k4*CC2H2);
            
        //----- precursor types for cracking
        const double xp = abs(tanh(5.73-0.00384*abs(T)-0.159*log10(Ntar))/6.0-0.218+0.0277*log10(Ntar));
        const double xn = abs(tanh(-1.98+6.18E-4*abs(T)+0.124*log10(Ntar)-0.00285*pow(log10(Ntar),2.0)+4.14E-7*pow(abs(T),2.0)-4.97E-5*abs(T)*log10(Ntar))/2.0-0.576+0.000233*abs(T)-1.69E-7*pow(abs(T),2.0));
        const double xt = abs(tanh(17.3-0.00869*abs(T)-1.08*log10(Ntar)+0.0199*pow(log10(Ntar),2.0)+0.000365*abs(T)*log10(Ntar))/3.0+0.000265*abs(T)-0.000111*pow(log10(Ntar),2.0)-9.32E-6*abs(T)*log10(Ntar));
        const double xb = 1-xp-xn-xt;
       
        //----- Compute change in tar 
        double Rsn, Rpd, temp_Rsn, temp_Rpd, Rpc, Rps, Rss, Rsc;
        double total_time,delta_time;
        double Ntar_old;
        double Ntar_new=0;

        Rsn = 0.0;
        Rpd = 0.0;
        total_time = 0.0;
        delta_time = dt;
        Ntar_old = Ntar;

        while (dt> total_time){
          if(Ntar_old<1.0) {
            temp_Rsn = 0.0;
            temp_Rpd = 0.0;
            Rpc = 0.0;
            Rps = 0.0;
          }
          else {
            temp_Rsn = 4*eps*pow(da,2)*pow(Ntar_old,2.0)*pow(2*M_PI*kB*T*mtar,1.0/2.0)/(3*MWc);
            temp_Rpd = (Nsoot > 0.0) ? eps*pow(kB*T,1.0/2.0)*(pow(mtar,1.0/2.0)*pow(2/M_PI,1.0/6.0)*pow(3*Msoot/(Nsoot*rhos),2.0/3.0)+
                       da*pow(3.0/MWc,1.0/2.0)*pow(M_PI,1.0/3.0)*pow(6.0*Msoot/(Nsoot*rhos),2.0/3.0)+
                       pow(da,2.0)*pow(2.0*M_PI*mtar/9,1.0/2.0))*Ntar_old*Nsoot : 0.0;
            Rpc = (31.1/94.0*1.0E7*exp(-1.0E8/(Rgas*T))*xp+
                  1.0E8*exp(-1.0E8/(Rgas*T))*xp+
                  50.0/128.0*1.58E12*exp(-3.24E8/(Rgas*T))*xn*pow(CH2/(Rgas*T),0.4)+
                  14.0/92.0*1.04E12*exp(-2.47E8/(Rgas*T))*xt*pow(CH2/(Rgas*T),0.5)+
                  4.4E8*exp(-2.2E8/(Rgas*T))*xb)*Ntar_old;
            Rps = 2508*Ntar_old*(HACAp-Oxidation-Gasification);
          }
          Ntar_new = Ntar_old + (-2*temp_Rsn-temp_Rpd-Rpc+Rps)*delta_time;

          //----- Test to see if we need a smaller timestep
          if (Ntar_new < 0.0){
            delta_time = delta_time/10.0;
          } else {
            Ntar_old   = Ntar_new;
            Rsn        = Rsn + temp_Rsn*delta_time/dt;
            Rpd        = Rpd + temp_Rpd*delta_time/dt;
            total_time = total_time+delta_time;
            delta_time = dt-total_time;
          }
         //----- End while loop
        }
        if (Ntar_new < 1.0e6) {
          tar_src[c] = -(Tar[c]*rho[c])/dt;
        } else {
          tar_src[c] = (Ntar_new-Ntar)*mtar/dt;
        }   
        
        //------ Compute change in soot
        double Kn,beta,SootAlpha,HACAs,dg,dp,lambdag;
        //double Msoot_new,Msoot_old,Nsoot_new,Nsoot_old,msoot;
        double Msoot_old,Nsoot_old,msoot;
        double Msoot_new=0.0;
        double Nsoot_new=0.0;
        total_time = 0.0;
        delta_time = dt;

        Msoot_old  = Msoot;
        Nsoot_old  = Nsoot;
        
        while (dt>total_time){
          msoot = Msoot_old /Nsoot_old; 

          if(Msoot_old<=0.0 or Nsoot_old<=0) {
            Rpd = 0.0;
            Rsc = 0.0;
            Rss = 0.0;
          } else{
            //------ Surface Reactions
            SootAlpha = abs(tanh(a/log10(msoot)+b));
            HACAs = k4*CC2H2*SootAlpha*chiC*(k1*CH+k2*COH)*mC2H2/(kr1*CH2+kr2*CH2O+k3*CH+k4*CC2H2);
            Rss = M_PI*pow(6.0*msoot/(M_PI*rhos),2.0/3.0)*Nsoot_old*(HACAs-Oxidation-Gasification);
            //------ Coagulation
            dg = pow(6.0*kB*T/(P*M_PI),1.0/3.0);
            dp = pow(6.0*msoot/(rhos*M_PI),1.0/3.0);
            lambdag = kB*T/(pow(2.0,0.5)*M_PI*pow(dg,2.0)*P);
            Kn = lambdag/dp;
            if(Kn<0.1){
              beta = eps*pow(dp,2.0)*pow(8.0*M_PI*kB*T/msoot,0.5);
            } else if(Kn>10.0){
              beta = 8.0*kB*T/(3.0*Visc)*(1.0+1.257*Kn);
            } else{
              beta = (8.0*kB*T/(3.0*Visc)*(1.0+1.257*Kn))/(1+Kn)+(eps*pow(dp,2.0)*pow(8.0*M_PI*kB*T/msoot,0.5))/(1+1/Kn);
            }
            Rsc = beta*pow(Nsoot_old,2.0);
          }

          Nsoot_new = Nsoot_old+(Rsn-Rsc)*delta_time;
          Msoot_new = Msoot_old+(2*mtar*Rsn+mtar*Rpd+Rss)*delta_time;
          
          //----- Test if we need to take a smaller time step
          if (Msoot_new <= 0.0){
            break;
          } else if (Nsoot_new <= 0.0){
            delta_time = Nsoot_old/Rsc/10.0;
          } else {
            Msoot_old  = Msoot_new;
            Nsoot_old  = Nsoot_new;
            total_time = total_time+delta_time;
            delta_time = dt-total_time;
          }
          //----- End while loop
        }
        if (Msoot_new > 0.0 and Nsoot_new > 1){
          num_density_src[c] = (Nsoot_new-Nsoot)/dt;
          soot_mass_src[c]   = (Msoot_new-Msoot)/dt;
        } else{
          num_density_src[c] = -(Nsoot)/dt;
          soot_mass_src[c]   = -(Msoot)/dt;
        }
        balance_src[c]     = ((Ntar-Ntar_new)*mtar+(Msoot-Msoot_new))/dt;
    }
  }
}


//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
MonoSoot::sched_initialize( const LevelP& level, SchedulerP& sched )
{
    string taskname = "MonoSoot::initialize";

    Task* tsk = scinew Task(taskname, this, &MonoSoot::initialize);

    tsk->computes(m_tar_src_label);
    tsk->computes(m_nd_src_label);
    tsk->computes(m_soot_mass_src_label);
    tsk->computes(m_balance_src_label);

    sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));}
void
MonoSoot::initialize( const ProcessorGroup* pc,
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

    CCVariable<double> tar_src;
    CCVariable<double> num_density_src;
    CCVariable<double> soot_mass_src;
    CCVariable<double> balance_src;

    new_dw->allocateAndPut( tar_src, m_tar_src_label, matlIndex, patch );
    new_dw->allocateAndPut( num_density_src, m_nd_src_label, matlIndex, patch );
    new_dw->allocateAndPut( soot_mass_src, m_soot_mass_src_label, matlIndex, patch );
    new_dw->allocateAndPut( balance_src, m_balance_src_label, matlIndex, patch );

    tar_src.initialize(0.0);
    num_density_src.initialize(0.0);
    soot_mass_src.initialize(0.0);
    balance_src.initialize(0.0);

  }
}
