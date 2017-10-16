#ifndef Uintah_Component_Arches_BOUNDARYFUNCTORS_h
#define Uintah_Component_Arches_BOUNDARYFUNCTORS_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Task/FieldContainer.h>
#include <CCA/Components/Arches/Task/TaskVariableTools.h>
#include <CCA/Components/Arches/BoundaryConditions/BoundaryFunctorHelper.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <CCA/Components/Arches/GridTools.h>

namespace Uintah { namespace ArchesCore{

  template <typename T>
  class BCFunctors {

  public:

    enum FUNCTOR_TYPE { DIRICHLET_FUN, NEUMANN_FUN };

    std::string func_enum_str( FUNCTOR_TYPE fun_type ){

      if ( fun_type == DIRICHLET_FUN ){
        return "dirichlet";
      } else if ( fun_type == NEUMANN_FUN ){
        return "nuemann";
      } else {
        throw InvalidValue("Error: BC Functor type not recognized.",__FILE__,__LINE__);
      }

    }

    BCFunctors(){}

    ~BCFunctors(){}


    std::string pair_face_var_names( std::string face_name, std::string var_name ){
      return face_name + "_" + var_name;
    }

    struct BaseFunctor;
    struct Dirichlet;
    struct Neumann;
    struct MassFlow;
    struct MMSalmgren;
    struct MMSshunn;
    struct SecondaryVariableBC;
    struct VelocityBC;

    void create_bcs( ProblemSpecP& db, std::vector<std::string> variables );

    std::shared_ptr<BaseFunctor> get_functor( BndCondTypeEnum bnd_type ){
      // This naming isn't great. Would be better to map these with a static function??
      std::string name;
      if ( bnd_type == DIRICHLET ){
        name = "Dirichlet";
      } else if ( bnd_type == NEUMANN ){
        name = "Neumann";
      }

       auto i = m_bcFunStorage.find(name);

       if ( i != m_bcFunStorage.end() ){
         return i->second;
       }

       throw InvalidValue("Error: Cannot locate BC functor with name: "+name, __FILE__, __LINE__);

    }

    void get_bc_dependencies( std::vector<std::string> varnames, WBCHelper* bc_helper,
                              std::vector<std::string>& dep );


    void apply_bc( std::vector<std::string> varnames, WBCHelper* bc_helper,
                   ArchesTaskInfoManager* tsk_info, const Patch* patch );

  private:

    void insert_functor( std::string name, std::shared_ptr<BaseFunctor> fun ){

      auto iter = m_bcFunStorage.find(name);
      if ( iter == m_bcFunStorage.end() ){
        m_bcFunStorage.insert(std::make_pair(name, fun));
      }

    }

    void insert_functor( std::string face_name, std::string var_name,
                         std::shared_ptr<BaseFunctor> fun ){

      std::string name = pair_face_var_names( face_name, var_name );
      auto iter = m_bcFunStorage.find(name);
      if ( iter == m_bcFunStorage.end() ){
        m_bcFunStorage.insert(std::make_pair(name, fun));
      }

    }

    std::map<std::string, std::shared_ptr<BaseFunctor> > m_bcFunStorage;

};

//--------------------------------------------------------------------------------------------------
// IMPLEMENTATION
//--------------------------------------------------------------------------------------------------

template <typename T>
void BCFunctors<T>::create_bcs( ProblemSpecP& db, std::vector<std::string> variables ){

  ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP db_bc = db_root->findBlock("Grid")->findBlock("BoundaryConditions");

  if ( db_bc ){

    for ( Uintah::ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != nullptr; db_face = db_face->findNextBlock("Face") ){

      std::string face_name = "NOT_NAMED";
      db_face->getAttribute( "name", face_name );

      if ( face_name == "NOT_NAMED" ){
        throw ProblemSetupException("Error: You must have a name attribute for all Face boundary conditions.", __FILE__, __LINE__);
      }

      for ( Uintah::ProblemSpecP db_bc_type = db_face->findBlock("BCType"); db_bc_type != nullptr; db_bc_type = db_bc_type->findNextBlock("BCType") ){

        std::string type;
        std::string varname;

        db_bc_type->getAttribute("var", type);
        db_bc_type->getAttribute("label", varname);

        bool matched_var = false;
        for ( auto i = variables.begin(); i != variables.end(); i++ ){
          if ( *i == varname ){
            matched_var = true;
          }
        }

// -------------- STANDARD BOUNDARY CONDITIONS -----------------------------------------------------
// Note: These are
//       If the names don't match, then this would be no bueno.

        if ( matched_var ){
          if ( type == "Dirichlet" ){

            std::shared_ptr<BaseFunctor> fun(scinew Dirichlet());
            insert_functor(func_enum_str(DIRICHLET_FUN), fun);

          } else if ( type == "Neumann" ){

            std::shared_ptr<BaseFunctor> fun(scinew Neumann());
            insert_functor(func_enum_str(NEUMANN_FUN), fun);

          } else if ( type == "Custom" ){

            std::string custom_type="NA";
            db_bc_type->getAttribute("type",custom_type);

// --------------- HERE IS WHERE WE INSERT CUSTOMIZABLE BC FUNCTORS --------------------------------

            if ( custom_type == "massflow" ){

              double mdot = 0.0;
              db_bc_type->require("value", mdot);

              std::string density_label = "density";
              if ( db_bc_type->findBlock("density") ){
                db_bc_type->findBlock("density")->getAttribute("label", density_label);
              }

              std::shared_ptr<BaseFunctor> fun(scinew MassFlow(mdot, density_label));
              insert_functor( face_name, varname, fun);
            } else if ( custom_type == "MMS_almgren" ) {

              std::string x_label = "gridX";
              std::string y_label = "gridY";
              std::string which_vel = "u";

              db_bc_type->require("which_vel", which_vel);
              ProblemSpecP db_coord =  db_bc_type->findBlock("coordinates");

              if ( db_coord ){
                db_coord->getAttribute("x", x_label);
                db_coord->getAttribute("y", y_label);
              } else {
                throw InvalidValue("Error: must have coordinates specified for almgren MMS init condition", __FILE__, __LINE__);
              }

              std::shared_ptr<BaseFunctor> fun(scinew MMSalmgren(x_label, y_label, which_vel));
              insert_functor( face_name, varname, fun);

            } else if ( custom_type == "MMS_shunn" ) {

              std::string x_label = "gridX";

              ProblemSpecP db_coord =  db_bc_type->findBlock("coordinates");

              if ( db_coord ){
                db_coord->getAttribute("x", x_label);
              } else {
                throw InvalidValue("Error: must have coordinates specified for shunn MMS init condition", __FILE__, __LINE__);
              }

              std::shared_ptr<BaseFunctor> fun(scinew MMSshunn(x_label));
              insert_functor( face_name, varname, fun);

            } else if ( custom_type == "table_value" ) {

              std::string tabulated_var_name = "NA";
              db_bc_type->require("value", tabulated_var_name);

              std::shared_ptr<BaseFunctor> fun(scinew SecondaryVariableBC(tabulated_var_name));
              insert_functor( face_name, varname, fun );

            } else if ( custom_type == "handoff" ) {

              std::string handoff_var_name = "NA";
              db_bc_type->require("value", handoff_var_name);

              std::shared_ptr<BaseFunctor> fun(scinew SecondaryVariableBC(handoff_var_name));
              insert_functor( face_name, varname, fun );

            } else if ( custom_type == "velocity" ){

              //HARD CODED! - Possibly fix for the future?
              std::string density_name = "density";

              double vel_value=0.0;
              db_bc_type->require("value", vel_value);

              std::shared_ptr<BaseFunctor> fun( scinew VelocityBC(density_name, vel_value));

              insert_functor( face_name, varname, fun );

            } else {

              throw InvalidValue("Error: Custom functor type not recognized", __FILE__, __LINE__);

            }

          } else {

            throw InvalidValue("Error: BC var type not recognized: "+type, __FILE__, __LINE__);

          }
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
// BASE FUNCTOR

template <typename T>
struct BCFunctors<T>::BaseFunctor{

public:

  BaseFunctor(){}
  virtual ~BaseFunctor(){}

  virtual void add_dep( std::vector<std::string>& master_dep ) = 0;
  virtual void eval_bc(
    std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
    const BndSpec* bnd, Uintah::Iterator bndIter ) = 0;

protected:

  void check_master_list( std::vector<std::string>& local_dep ,
    std::vector<std::string>& master_list ){

    for ( auto ilocal = local_dep.begin(); ilocal != local_dep.end(); ilocal++ ){

      auto imaster = std::find( master_list.begin(), master_list.end(), *ilocal );
      if ( imaster == master_list.end() ){
        master_list.push_back( *ilocal );
      }

    }
  }

};

//--------------------------------------------------------------------------------------------------
// DERIVED FUNCTORS

template <typename T>
struct BCFunctors<T>::Dirichlet : BaseFunctor{
public:
  Dirichlet(){}
  ~Dirichlet(){}

  void add_dep( std::vector<std::string>& master_dep ){}

  void eval_bc( std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::Iterator bndIter ){

    VariableHelper<T> var_help;
    T& var = *( tsk_info->get_uintah_field<T>(var_name));
    IntVector iDir = patch->faceDirection( bnd->face );
    IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);

    const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];

    const BndCondSpec* spec = bnd->find(var_name);

    if ( var_help.dir == ArchesCore::NODIR || dot == 0){

      // CCVariable or CC position in the staggered variable
      for ( bndIter.reset(); !bndIter.done(); bndIter++ ){

        var[*bndIter] = 2.0 * spec->value - var[*bndIter-iDir];

      }

    } else {

      // Staggered Variable
      if ( dot == -1 ){
      // Normal face -
        for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
          var[*bndIter] = spec->value;
          var[*bndIter - iDir] = spec->value;
        }
      } else if ( dot == 1 ){
      // Normal face +
        for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
          var[*bndIter] = spec->value;
        }
      } else {
          throw InvalidValue("Error: ...",__FILE__,__LINE__);
      }
    }
  }

private:
  std::vector<std::string> m_dep;

};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::Neumann: BaseFunctor{
public:
  Neumann(){}
  ~Neumann(){}

  void add_dep( std::vector<std::string>& master_dep ){}

  void eval_bc( std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::Iterator bndIter ){

    VariableHelper<T> var_help;
    T& var = *( tsk_info->get_uintah_field<T>(var_name));
    IntVector iDir = patch->faceDirection( bnd->face );
    IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);
    Vector Dx = patch->dCell();
    double dx = 0.;
    const double norm = iDir[0]+iDir[1]+iDir[2];
    const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];

    if ( var_help.dir == ArchesCore::NODIR || dot == 0  ){
    // CCvariables and CC positions in a staggered variable

      IntVector normIDir = iDir * iDir;
      if ( normIDir == IntVector(1,0,0) ){
        dx = Dx.x();
      } else if ( normIDir == IntVector(0,1,0) ){
        dx = Dx.y();
      } else {
        dx = Dx.z();
      }

      const BndCondSpec* spec = bnd->find(var_name);

      for ( bndIter.reset(); !bndIter.done(); bndIter++ ){

        var[*bndIter] = norm*dx * spec->value + var[*bndIter-iDir];

      }
    } else {

      // for staggered variables, only going to allow for zero gradient, one-sided for now

      if ( dot == -1 ){
        for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
          IntVector two_iDir(iDir[0]*2,iDir[1]*2,iDir[2]*2);
          var[*bndIter - iDir] = var[*bndIter - two_iDir];
          var[*bndIter] = var[*bndIter-iDir];
        }
      } else if ( dot == 1 ){
        for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
          var[*bndIter] = var[*bndIter-iDir];
        }
      }

    }
  }

private:
  std::vector<std::string> m_dep;

};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::MassFlow : BaseFunctor{

public:

  MassFlow( const double mdot, std::string density_name ) : BaseFunctor(), m_mdot(mdot),
    m_density_name(density_name) {}
  ~MassFlow(){}

  void add_dep( std::vector<std::string>& master_dep ){

    m_dep.push_back( m_density_name );
    BaseFunctor::check_master_list( m_dep, master_dep );

  }

  void eval_bc( std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::Iterator bndIter ){

    T& var = *( tsk_info->get_uintah_field<T>(var_name));
    constCCVariable<double> rho =
      *( tsk_info->get_const_uintah_field<constCCVariable<double> >(m_density_name));

    VariableHelper<T> var_help;
    IntVector iDir = patch->faceDirection( bnd->face );
    IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);

    const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];

    if ( dot == -1 ){
    // Normal face (-)  staggered variables
      for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
        const double rho_face = 0.5*(rho[*bndIter] + rho[*bndIter - iDir]);
        var[*bndIter] = m_mdot / (rho_face * bnd->area );
        var[*bndIter - iDir] = var[*bndIter];
      }
    } else if ( dot == 1 ){
    // Normal face (+)  staggered variables
      for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
        double rho_face = 0.5*(rho[*bndIter] + rho[*bndIter - iDir]);
        var[*bndIter] = m_mdot / (rho_face * bnd->area );
      }
    } else {
      throw InvalidValue("Error: Trying to set a massflow rate for a non-normal velocoty", __FILE__, __LINE__);
    }
  }

private:

  const double m_mdot;
  std::string m_density_name;

  std::vector<std::string> m_dep;


};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::MMSalmgren : BaseFunctor{

public:

  MMSalmgren( std::string x_name, std::string y_name, std::string which_vel) : BaseFunctor(), m_x_name(x_name),
    m_y_name(y_name), m_which_vel(which_vel){}
  ~MMSalmgren(){}

  void add_dep( std::vector<std::string>& master_dep ){

    m_dep.push_back( m_y_name );
    m_dep.push_back( m_x_name );
    BaseFunctor::check_master_list( m_dep, master_dep );

  }

  void eval_bc( std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::Iterator bndIter ){

    double m_two_pi = 2.0*acos(-1.0);
    double m_amp = 1.0;

    T& var = *( tsk_info->get_uintah_field<T>(var_name));

    constCCVariable<double> x =
      *( tsk_info->get_const_uintah_field<constCCVariable<double> >(m_x_name));

    constCCVariable<double> y =
      *( tsk_info->get_const_uintah_field<constCCVariable<double> >(m_y_name));

    VariableHelper<T> var_help;
    IntVector iDir = patch->faceDirection( bnd->face );
    IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);

    const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];


    if ( m_which_vel == "u" ){

      if (var_help.dir == ArchesCore::NODIR ){
        // scalar or CCvariable
        for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
          var[*bndIter] = 1.0  - m_amp * cos( m_two_pi * x[*bndIter] )
                                  * sin( m_two_pi * y[*bndIter] );
        }
      } else {
        // SFCX or SFCY or SFCZ variable
          if (dot == -1){
            for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
              // (-) faces  two cells at the begin of the domain
              // value are the same because we do not resolve extra cell (i=-1), and we set the BC at i = 0
//                    var[*bndIter] = 1.0  - m_amp * cos( m_two_pi * x[*bndIter] )
//                                      * sin( m_two_pi * y[*bndIter] );

                var[*bndIter] = 1.0  - m_amp * cos( m_two_pi * x[*bndIter- iDir] )
                                  * sin( m_two_pi * y[*bndIter- iDir] );


                 var[*bndIter- iDir] = 1.0  - m_amp * cos( m_two_pi * x[*bndIter - iDir] )
                                  * sin( m_two_pi * y[*bndIter- iDir] );
            }
          } else {
             for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
               // (+) faces one cell at the end of the domain
               var[*bndIter] = 1.0  - m_amp * cos( m_two_pi * x[*bndIter] )
                                  * sin( m_two_pi * y[*bndIter] );
              }
          }
      }
    } else if ( m_which_vel == "v" ){

      if (var_help.dir == ArchesCore::NODIR ){
        // scalar
        for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
          var[*bndIter] = 1.0  + m_amp * sin( m_two_pi * x[*bndIter] )
                              * cos( m_two_pi * y[*bndIter] );
        }
      } else {
      // SFCX or SFCY or SFCZ variable
        if (dot == -1){
          for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
            // (-) faces  two cells
//                var[*bndIter] = 1.0  + m_amp * sin( m_two_pi * x[*bndIter] )
//                                      * cos( m_two_pi * y[*bndIter] );
            var[*bndIter] = 1.0  + m_amp * sin( m_two_pi * x[*bndIter- iDir] )
                   * cos( m_two_pi * y[*bndIter- iDir] );


            var[*bndIter- iDir] = 1.0  + m_amp * sin( m_two_pi * x[*bndIter- iDir] )
                                  * cos( m_two_pi * y[*bndIter- iDir] );
          }
        } else {
        for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
          // (+) faces
          var[*bndIter] = 1.0  + m_amp * sin( m_two_pi * x[*bndIter] )
                                  * cos( m_two_pi * y[*bndIter] );

        }
      }
  }
    } else {
      throw InvalidValue("Error: Almgren BC  velocoty does not exit", __FILE__, __LINE__);
    }
    }
private:

  std::string m_x_name;
  std::string m_y_name;
  std::string m_which_vel;
  std::vector<std::string> m_dep;


};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::MMSshunn : BaseFunctor{

public:

  MMSshunn( std::string x_name) : BaseFunctor(), m_x_name(x_name) {}
  ~MMSshunn(){}

  void add_dep( std::vector<std::string>& master_dep ){

    m_dep.push_back( m_x_name );
    BaseFunctor::check_master_list( m_dep, master_dep );

  }

  void eval_bc( std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::Iterator bndIter ){

    //const double m_two_pi = 2.0*acos(-1.0);
    //const double m_amp = 1.0;
    const double m_k1  = 4.0;
    const double m_k2  = 2.0;
    const double m_w0  = 50.0;
    const double m_rho0 = 20.0;
    const double m_rho1 = 1.0;

    double time_d  = tsk_info->get_time();
    int   time_substep = tsk_info->get_time_substep();
    double factor      = tsk_info->get_ssp_time_factor(time_substep);
    double dt          = tsk_info->get_dt();
    time_d = time_d + factor*dt;


    T& var = *( tsk_info->get_uintah_field<T>(var_name));

    constCCVariable<double> x =
      *( tsk_info->get_const_uintah_field<constCCVariable<double> >(m_x_name));

    VariableHelper<T> var_help;
    //IntVector iDir = patch->faceDirection( bnd->face );
    //IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);

    //const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];
    const double z1 = std::exp(-m_k1 * time_d);
    if (var_help.dir == ArchesCore::NODIR ){
      // scalar or CCvariable
      for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
        const double z2 = std::cosh(m_w0 * std::exp (-m_k2 * time_d) *x[*bndIter]); // x is cc value
        const double phi = (z1-z2)/(z1 * (1.0 - m_rho0/m_rho1)-z2);
        const double rho = 1.0/(phi/m_rho1 + (1.0- phi )/m_rho0);
        var[*bndIter] = phi*rho;
       }
     }


    }
private:

  std::string m_x_name;
  std::vector<std::string> m_dep;



};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::SecondaryVariableBC : BaseFunctor {

public:

  SecondaryVariableBC( std::string sec_var_name ) : m_sec_var_name(sec_var_name){
    m_dep.push_back( m_sec_var_name );
  }
  ~SecondaryVariableBC(){}

  void add_dep( std::vector<std::string>& master_dep ){

    // Now adding dependencies to the master list.
    // This checks for repeats to ensure a variable isn't added twice.
    BaseFunctor::check_master_list( m_dep, master_dep );

  }

  void eval_bc( std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::Iterator bndIter ){

    VariableHelper<T> var_help;
    typedef typename VariableHelper<T>::ConstType CT;
    T& var = *( tsk_info->get_uintah_field<T>(var_name));
    IntVector iDir = patch->faceDirection( bnd->face );
    CT& sec_var = *( tsk_info->get_const_uintah_field<CT>(m_sec_var_name));

    for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
      double interp_sec_var = (sec_var[*bndIter] + sec_var[*bndIter - iDir])/2.;
      var[*bndIter] = 2. * interp_sec_var - var[*bndIter - iDir];
    }

  }

private:

  std::string m_sec_var_name;
  std::vector<std::string> m_dep;
  SimulationStateP m_shared_state;

};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::VelocityBC : BaseFunctor {

public:

  VelocityBC( std::string density_name, double value ):m_density_name(density_name), m_vel_value(value){
    m_dep.push_back( m_density_name );
  }
  ~VelocityBC(){}

  void add_dep( std::vector<std::string>& master_dep ){

    // Now adding dependencies to the master list.
    // This checks for repeats to ensure a variable isn't added twice.
    BaseFunctor::check_master_list( m_dep, master_dep );

  }

  void eval_bc( std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::Iterator bndIter ){

    VariableHelper<T> var_help;
    //typedef typename VariableHelper<T>::ConstType CT;
    T& var = *( tsk_info->get_uintah_field<T>(var_name));
    constCCVariable<double>& rho =
      *( tsk_info->get_const_uintah_field<constCCVariable<double> >(m_density_name));

    IntVector iDir = patch->faceDirection( bnd->face );

    //
    IntVector offset(0,0,0);
    bool parallel_dir = false;
    if ( bnd->face == Patch::xminus || bnd->face == Patch::xplus ){

      if ( var_help.dir == XDIR ){
        parallel_dir = true;
      } else if ( var_help.dir == YDIR ){
        offset[1] = 1;
      } else if ( var_help.dir == ZDIR ){
        offset[2] = 1;
      }

    } else if ( bnd->face == Patch::yminus || bnd->face == Patch::yplus ){

      if ( var_help.dir == XDIR ){
        offset[0] = 1;
      } else if ( var_help.dir == YDIR ){
        parallel_dir = true;
      } else if ( var_help.dir == ZDIR ){
        offset[2] = 1;
      }

    } else if ( bnd->face == Patch::zminus || bnd->face == Patch::zplus ){

      if ( var_help.dir == XDIR ){
        offset[0] = 1;
      } else if ( var_help.dir == YDIR ){
        offset[1] = 1;
      } else if ( var_help.dir == ZDIR ){
        parallel_dir = true;
      }

    }

    IntVector offset_iDir = iDir + offset;

    IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);
    const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];


    if ( parallel_dir ){
        //The face normal and the velocity are in parallel
        if (dot == -1) {
            //Face +
            for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
                  const double interp_rho = 0.5*(rho[*bndIter-iDir]+rho[*bndIter]);
                 var[*bndIter-iDir] = interp_rho*m_vel_value;
                  var[*bndIter] = var[*bndIter-iDir];
            }
        } else {
            // Face -
            for ( bndIter.reset(); !bndIter.done(); bndIter++ ){
                  const double interp_rho = 0.5*(rho[*bndIter-iDir]+rho[*bndIter]);
                  var[*bndIter] = interp_rho*m_vel_value;
                  //var[*bndIter] = var[*bndIter-iDir];
        }

        }

    } else {
      //The face normal and the velocity are tangential
      for ( bndIter.reset(); !bndIter.done(); bndIter++ ){

        const double interp_rho =0.5*(0.5*(rho[*bndIter-iDir]+rho[*bndIter]) +
                                      0.5*(rho[*bndIter-offset_iDir]+rho[*bndIter + offset]));

        var[*bndIter] = 2.*interp_rho*m_vel_value - var[*bndIter-iDir];

      }
    }
  }

private:

  std::string m_density_name;
  const double m_vel_value;
  std::vector<std::string> m_dep;

};

// <put new functors here>

//--------- END BC FUNCTORS --------------

//--------------------------------------------------------------------------------------------------
template <typename T>
void BCFunctors<T>::get_bc_dependencies( std::vector<std::string> varnames, WBCHelper* bc_helper,
                          std::vector<std::string>& dep ){

  const BndMapT& bc_info = bc_helper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    std::string facename = i_bc->second.name;

    for ( auto i_eqn = varnames.begin(); i_eqn != varnames.end(); i_eqn++ ){

      const BndCondSpec* spec = i_bc->second.find(*i_eqn);

      if ( spec == NULL ){

        // Only throwing an error for edge BCs.
        // Interior BC's allowed to be missing
        if ( i_bc->second.edge_type == EDGE ){
          std::stringstream msg;
          msg << "Error: Cannot find a boundary condition for variable: " << *i_eqn << " on face: " << facename << std::endl;
          throw InvalidValue(msg.str(), __FILE__, __LINE__);
        }
        
      } else {

        std::shared_ptr<BaseFunctor> bc_fun = NULL;
        if ( spec->bcType == CUSTOM ){
          //CUSTOM BCS (i.e., NOT Dirichlet or Neumann)
          std::string key_name = pair_face_var_names( facename, *i_eqn );
          bc_fun = m_bcFunStorage[key_name];
        }

        if ( bc_fun != NULL ){
          bc_fun->add_dep( dep );
        }

      }
    }
  }
}

//--------------------------------------------------------------------------------------------------

// This function actually applies the BC to the variable(s)
template <typename T>
void BCFunctors<T>::apply_bc( std::vector<std::string> varnames, WBCHelper* bc_helper,
               ArchesTaskInfoManager* tsk_info, const Patch* patch ){

  const BndMapT& bc_info = bc_helper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    //Get the iterator
    Uintah::Iterator cell_iter = bc_helper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());

    std::string facename = i_bc->second.name;

    for ( auto i_eqn = varnames.begin(); i_eqn != varnames.end(); i_eqn++ ){

      const BndCondSpec* spec = i_bc->second.find(*i_eqn);

      if ( spec == NULL ){

        // Only throwing an error for edge BCs.
        // Interior BC's allowed to be missing
        if ( i_bc->second.edge_type == EDGE ){
          std::stringstream msg;
          msg << "Error: Cannot find a boundary condition for variable: " << *i_eqn << " on face: " << facename << std::endl;
          throw InvalidValue(msg.str(), __FILE__, __LINE__);
        }

      } else {

        std::shared_ptr<BaseFunctor> bc_fun = NULL;
        if ( spec->bcType == DIRICHLET ){
          bc_fun = m_bcFunStorage[func_enum_str(DIRICHLET_FUN)];
        } else if ( spec->bcType == NEUMANN ){
          bc_fun = m_bcFunStorage[func_enum_str(NEUMANN_FUN)];
        } else {
          //CUSTOM BCS
          std::string key_name = pair_face_var_names( facename, *i_eqn );
          bc_fun = m_bcFunStorage[key_name];
        }

        const BndSpec bndSpec = i_bc->second;

        // Actually applying the boundary condition here:
        if ( bc_fun != NULL ){
          bc_fun->eval_bc( *i_eqn, patch, tsk_info, &bndSpec, cell_iter );
        } else {
          throw InvalidValue(
            "Error: Boundary condition implementation not found for: "+*i_eqn, __FILE__, __LINE__);
        }
      }
    }
  }
}

} } //end namespace Uintah::BoundaryFunctor


#endif
