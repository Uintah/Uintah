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

static Uintah::DebugStream dbgbc("ARCHES_BC_FUNCTORS", false);
#define DBC_BC_ON  dbgbc.active()
#define DBGBCFUN  if( DBC_BC_ON  ) dbgbc

namespace Uintah { namespace ArchesCore{

  struct DepBCInfo {
    std::string variable_name;
    int n_ghosts{0};
    ArchesFieldContainer::WHICH_DW dw{ArchesFieldContainer::NEWDW};
  };

  typedef std::vector<DepBCInfo> FunctorDepList;

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
    struct PressureOutletBC;
    struct SubGridInjector;

    void create_bcs( ProblemSpecP& db, std::vector<std::string> variables );

    void build_bcs( ProblemSpecP db_bc, const std::vector<std::string> variables, const std::string );

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

    void get_bc_dependencies( std::vector<std::string>& varnames, WBCHelper* bc_helper,
                              FunctorDepList& dep );

    void get_bc_modifies( std::vector<std::string>& varnames, WBCHelper* bc_helper,
                          std::vector<std::string>& mod );


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

    DBGBCFUN << " BC Functors: ** Parsing Face boundary conditions. **" << std::endl;
    build_bcs( db_bc, variables, "Face" );
    DBGBCFUN << " BC Functors: ** Parsing InteriorFace boundary conditions. **" << std::endl;
    build_bcs( db_bc, variables, "InteriorFace" );
    DBGBCFUN << " BC Functors: ** Done parsing for boundary condition functors. **" << std::endl;

  }
}

template <typename T>
void BCFunctors<T>::build_bcs( ProblemSpecP db_bc, const std::vector<std::string> variables, const std::string EdgeType ){

 for ( Uintah::ProblemSpecP db_face = db_bc->findBlock(EdgeType); db_face != nullptr; db_face = db_face->findNextBlock(EdgeType) ){

   std::string face_name = "NOT_NAMED";
   db_face->getAttribute( "name", face_name );

   if ( face_name == "NOT_NAMED" ){
     throw ProblemSetupException("Error: You must have a name attribute for all "+EdgeType+" boundary conditions.", __FILE__, __LINE__);
   }

   for ( Uintah::ProblemSpecP db_bc_type = db_face->findBlock("BCType"); db_bc_type != nullptr; db_bc_type = db_bc_type->findNextBlock("BCType") ){

     std::string type;
     std::string varname;

     db_bc_type->getAttribute("var", type);
     db_bc_type->getAttribute("label", varname);

     //DBGBCFUN << " BC Functors: Looking for a boundary condition for variable " << varname << std::endl;

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

       DBGBCFUN << " BC Functors: Found a boundary condition for variable " << varname << std::endl;

       if ( type == "Dirichlet" ){

         std::shared_ptr<BaseFunctor> fun(scinew Dirichlet());
         insert_functor(func_enum_str(DIRICHLET_FUN), fun);

         DBGBCFUN << "              with type: " << type << std::endl;

       } else if ( type == "Neumann" ){

         std::shared_ptr<BaseFunctor> fun(scinew Neumann());
         insert_functor(func_enum_str(NEUMANN_FUN), fun);

         DBGBCFUN << "              with type: " << type << std::endl;

       } else if ( type == "Custom" ){

         std::string custom_type="NA";
         db_bc_type->getAttribute("type",custom_type);

         DBGBCFUN << "              with custom type: " << custom_type << std::endl;

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

         } else if ( custom_type == "PressureOutlet" ){
           // HARD CODED! - fix me 
           std::string vel_name = "uVel";
           if (varname == "x-mom"){
             vel_name = "uVel";
             //vel_name = "x-mom";
           } else if (varname == "y-mom") {
             vel_name = "vVel";
             //vel_name = "y-mom";
           } else if  (varname == "z-mom") {
             vel_name = "wVel";
             //vel_name = "z-mom";
           } else {
             throw InvalidValue("Error: can not find velocity name "+type, __FILE__, __LINE__);
           } 
           double vel_value=0.0;
           db_bc_type->require("value", vel_value);

           std::shared_ptr<BaseFunctor> fun( scinew PressureOutletBC(vel_name, vel_value));

           insert_functor( face_name, varname, fun );


         } else if ( custom_type == "subgrid_injector" ){

           std::shared_ptr<BaseFunctor> fun( scinew SubGridInjector(varname, db_bc_type));

           insert_functor( face_name, varname, fun );

         } else {

           throw InvalidValue("Error: Custom functor type not recognized: "+custom_type,
                              __FILE__, __LINE__);

         }

       } else {

         throw InvalidValue("Error: BC var type not recognized: "+type, __FILE__, __LINE__);

       }
     }
   }
 }
}

//--------------------------------------------------------------------------------------------------
// BASE FUNCTOR
template<typename  ExecutionSpace,typename MemorySpace> // THIS IS A PLACEHOLDER DEFINITION, It should be removed upon merging of the kokkos branch
using ExecutionObject = std::iterator<ExecutionSpace,MemorySpace>;

template <typename T>
struct BCFunctors<T>::BaseFunctor {

public:

  BaseFunctor(){}
  virtual ~BaseFunctor(){}

  /** @brief Add dependancies onto the Arches Task for the proper execution of the boundary
             condition. This may be a nullptr op.**/
  virtual void add_dep( FunctorDepList& master_dep ) = 0;
  /** @brief Add additional modifies onto the Arches Task (outside the root variable) for
             proper execution of the boundary condition. This may be a nullptr op. **/
  virtual void add_mod( std::vector<std::string>& master_mod ) = 0;
  /** @brief Actually evaluate the boundary condition **/
    template <typename ExecutionSpace, typename MemorySpace>
    void eval_bc( ExecutionObject<ExecutionSpace,MemorySpace> executionObject,
    std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
    const BndSpec* bnd, Uintah::ListOfCellsIterator& bndIter ) {
    if (BCFunctors<T>::Dirichlet*             child   = dynamic_cast<BCFunctors<T>::Dirichlet*           >(this)){
      child->eval_bc(executionObject,var_name,patch,tsk_info,bnd,bndIter);
    }else if (BCFunctors<T>::Neumann*               child   = dynamic_cast<BCFunctors<T>::Neumann*             >(this)){
      child->eval_bc(executionObject,var_name,patch,tsk_info,bnd,bndIter);
    }else if (BCFunctors<T>::MassFlow*              child   = dynamic_cast<BCFunctors<T>::MassFlow*            >(this)){
      child->eval_bc(executionObject,var_name,patch,tsk_info,bnd,bndIter);
    }else if (BCFunctors<T>::MMSalmgren*            child   = dynamic_cast<BCFunctors<T>::MMSalmgren*          >(this)){
      child->eval_bc(executionObject,var_name,patch,tsk_info,bnd,bndIter);
    }else if (BCFunctors<T>::MMSshunn*              child   = dynamic_cast<BCFunctors<T>::MMSshunn*            >(this)){
      child->eval_bc(executionObject,var_name,patch,tsk_info,bnd,bndIter);
    }else if (BCFunctors<T>::SecondaryVariableBC*   child   = dynamic_cast<BCFunctors<T>::SecondaryVariableBC* >(this)){
      child->eval_bc(executionObject,var_name,patch,tsk_info,bnd,bndIter);
    }else if (BCFunctors<T>::VelocityBC*            child   = dynamic_cast<BCFunctors<T>::VelocityBC*          >(this)){
      child->eval_bc(executionObject,var_name,patch,tsk_info,bnd,bndIter);
    }else if (BCFunctors<T>::PressureOutletBC*      child   = dynamic_cast<BCFunctors<T>::PressureOutletBC*          >(this)){
      child->eval_bc(executionObject,var_name,patch,tsk_info,bnd,bndIter);
    }else if (BCFunctors<T>::SubGridInjector*       chile   = dynamic_cast<BCFunctors<T>::SubGridInjector*     >(this)){
      child->eval_bc(executionObject,var_name,patch,tsk_info,bnd,bndIter);
    }else {
          throw InvalidValue("Portable Boundary condition not properly included in source code.",__FILE__,__LINE__);
   }
}

protected:

  FunctorDepList m_dep;
  std::vector<std::string> m_mod;

  /** @brief Check the master list of all dependencies for the requirements of this specific bc functor **/
  void check_master_list( FunctorDepList& local_dep ,
    FunctorDepList& master_list ){

    for ( auto ilocal = local_dep.begin(); ilocal != local_dep.end(); ilocal++ ){

      bool found_match = false;
      for ( auto imaster = master_list.begin(); imaster != master_list.end(); imaster++ ){
        if ( ((*ilocal).variable_name == (*imaster).variable_name) && ((*ilocal).dw == (*imaster).dw) ){
          found_match = true;
        }
      }

      if ( !found_match ){
        master_list.push_back(*ilocal);
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

  void add_dep( FunctorDepList& master_dep ){}

  void add_mod( std::vector<std::string>& master_mod ){}

  template <typename ES, typename MS>
  void eval_bc( ExecutionObject<ES,MS>& executionObject,std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::ListOfCellsIterator& bndIter ){

    VariableHelper<T> var_help;
    T& var = *( tsk_info->get_uintah_field<T>(var_name));
    const IntVector iDir = patch->faceDirection( bnd->face );
    const IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);

    const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];

    const BndCondSpec* spec = bnd->find(var_name);

    const double spec_value=spec->value;

    if ( var_help.dir == ArchesCore::NODIR || dot == 0){


      // CCVariable or CC position in the staggered variable
      parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
        int im=i - iDir[0];
        int jm=j - iDir[1];
        int km=k - iDir[2];
        var(i,j,k) = 2.0 * spec_value - var(im,jm,km);
      });

    } else {

      // Staggered Variable
      if ( dot == -1 ){
      // Normal face -
      parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
          int im=i - iDir[0];
          int jm=j - iDir[1];
          int km=k - iDir[2];
          var(i,j,k) = spec_value;
          var(im,jm,km) = spec_value;
        });
      } else if ( dot == 1 ){
      // Normal face +
      parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
          var(i,j,k) = spec_value;
        });
      } else {
          throw InvalidValue("Error: ...",__FILE__,__LINE__);
      }
    }
  }

private:

};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::Neumann: BaseFunctor{
public:
  Neumann(){}
  ~Neumann(){}

  void add_dep( FunctorDepList& master_dep ){}

  void add_mod( std::vector<std::string>& master_mod ){}

template <typename ES, typename MS>
  void eval_bc( ExecutionObject<ES,MS>& executionObject,std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::ListOfCellsIterator& bndIter  ){

    VariableHelper<T> var_help;
    T& var = *( tsk_info->get_uintah_field<T>(var_name));
    const IntVector iDir = patch->faceDirection( bnd->face );
    const IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);
    const Vector Dx = patch->dCell();
    const double norm = iDir[0]+iDir[1]+iDir[2];
    const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];

    if ( var_help.dir == ArchesCore::NODIR || dot == 0  ){
    // CCvariables and CC positions in a staggered variable

      const IntVector normIDir = iDir * iDir;
      const double dx = normIDir == IntVector(1,0,0) ? Dx.x() : normIDir == IntVector(0,1,0) ? Dx.y() : Dx.z() ;

      const double spec_value = bnd->find(var_name)->value;

      parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
        int im = i - iDir[0];
        int jm = j - iDir[1];
        int km = k - iDir[2];

        var(i,j,k) = norm*dx * spec_value + var(im,jm,km);

      });
    } else {

      // for staggered variables, only going to allow for zero gradient, one-sided for now
      if ( dot == -1 ){
      parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
          int im=i - iDir[0];
          int jm=j - iDir[1];
          int km=k - iDir[2];
          int imm=i - 2*iDir[0];
          int jmm=j - 2*iDir[1];
          int kmm=k - 2*iDir[2];

          var(im,jm,km) = var(imm,jmm,kmm);
          var(i,j,k) = var(im,jm,km);
        });
      } else if ( dot == 1 ){
      parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
          int im=i - iDir[0];
          int jm=j - iDir[1];
          int km=k - iDir[2];
          var(i,j,k) = var(im,jm,km);
        });
      }

    }
  }

private:

};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::MassFlow : BaseFunctor{

public:

  MassFlow( const double mdot, std::string density_name ) : BaseFunctor(), m_mdot(mdot),
    m_density_name(density_name) {}
  ~MassFlow(){}

  void add_dep( FunctorDepList& master_dep ){

    // Note that the brace initializer won't work with compilers < c++14 because
    // of the default values set in the struct for n_ghosts and dw.
    DepBCInfo density_info;
    density_info.variable_name = m_density_name;

    BaseFunctor::m_dep.push_back( density_info );

    BaseFunctor::check_master_list( BaseFunctor::m_dep, master_dep );

  }

  void add_mod( std::vector<std::string>& master_mod ){}

template <typename ES, typename MS>
  void eval_bc( ExecutionObject<ES,MS>& executionObject,std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::ListOfCellsIterator& bndIter  ){

    T& var = *( tsk_info->get_uintah_field<T>(var_name));
    constCCVariable<double> rho =
      *( tsk_info->get_const_uintah_field<constCCVariable<double> >(m_density_name));

    VariableHelper<T> var_help;
    const IntVector iDir = patch->faceDirection( bnd->face );
    const IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);

    const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];

    const double bound_area = bnd->area;
    if ( dot == -1 ){
    // Normal face (-)  staggered variables
      parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
          int im=i - iDir[0];
          int jm=j - iDir[1];
          int km=k - iDir[2];
          const double rho_face = 0.5*(rho(i,j,k)+ rho(im,jm,km));
          var(i,j,k) = m_mdot / (rho_face * bound_area);
          var(im,jm,km) = var(i,j,k);
      });
    } else if ( dot == 1 ){
    // Normal face (+)  staggered variables
      parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
          int im=i - iDir[0];
          int jm=j - iDir[1];
          int km=k - iDir[2];
          double rho_face = 0.5*(rho(i,j,k) + rho(im,jm,km));
          var[*bndIter] = m_mdot / (rho_face * bound_area );
      });
    } else {
      throw InvalidValue("Error: Trying to set a massflow rate for a non-normal velocoty", __FILE__, __LINE__);
    }
  }

private:

  const double m_mdot;
  std::string m_density_name;

};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::MMSalmgren : BaseFunctor{

public:

  MMSalmgren( std::string x_name, std::string y_name, std::string which_vel) : BaseFunctor(), m_x_name(x_name),
    m_y_name(y_name), m_which_vel(which_vel){}
  ~MMSalmgren(){}

  void add_dep( FunctorDepList& master_dep ){

    DepBCInfo y_info; y_info.variable_name = m_y_name;
    DepBCInfo x_info; x_info.variable_name = m_x_name;
    BaseFunctor::m_dep.push_back( y_info );
    BaseFunctor::m_dep.push_back( x_info );
    BaseFunctor::check_master_list( BaseFunctor::m_dep, master_dep );

  }

  void add_mod( std::vector<std::string>& master_mod ){}

template <typename ES, typename MS>
  void eval_bc( ExecutionObject<ES,MS>& executionObject,std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::ListOfCellsIterator& bndIter  ){

    const double m_two_pi = 2.0*acos(-1.0);
    const double m_amp = 1.0;

    T& var = *( tsk_info->get_uintah_field<T>(var_name));

    constCCVariable<double> x =
      *( tsk_info->get_const_uintah_field<constCCVariable<double> >(m_x_name));

    constCCVariable<double> y =
      *( tsk_info->get_const_uintah_field<constCCVariable<double> >(m_y_name));

    VariableHelper<T> var_help;
    const IntVector iDir = patch->faceDirection( bnd->face );
    const IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);

    const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];


    if ( m_which_vel == "u" ){

      if (var_help.dir == ArchesCore::NODIR ){
        // scalar or CCvariable
        parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
          var(i,j,k) = 1.0  - m_amp * cos( m_two_pi * x(i,j,k) )
                                  * sin( m_two_pi * y(i,j,k) );
        });
      } else {
        // SFCX or SFCY or SFCZ variable
          if (dot == -1){
        parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
              // (-) faces  two cells at the begin of the domain
              // value are the same because we do not resolve extra cell (i=-1), and we set the BC at i = 0
//                    var[*bndIter] = 1.0  - m_amp * cos( m_two_pi * x[*bndIter] )
//                                      * sin( m_two_pi * y[*bndIter] );
                int im=i - iDir[0];
                int jm=j - iDir[1];
                int km=k - iDir[2];

                var(i,j,k) = 1.0  - m_amp * cos( m_two_pi * x(im,jm,km) )
                                  * sin( m_two_pi * y(im,jm,km) );


                 var(im,jm,km) = 1.0  - m_amp * cos( m_two_pi * x(im,jm,km) )
                                  * sin( m_two_pi * y(im,jm,km) );
            });
          } else {
        parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
               // (+) faces one cell at the end of the domain
               var(i,j,k) = 1.0  - m_amp * cos( m_two_pi * x(i,j,k) )
                                  * sin( m_two_pi * y(i,j,k) );
              });
          }
      }
    } else if ( m_which_vel == "v" ){

      if (var_help.dir == ArchesCore::NODIR ){
        // scalar
        parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
          var(i,j,k) = 1.0  + m_amp * sin( m_two_pi * x(i,j,k) )
                              * cos( m_two_pi * y(i,j,k) );
        });
      } else {
      // SFCX or SFCY or SFCZ variable
        if (dot == -1){
        parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
          int im=i - iDir[0];
          int jm=j - iDir[1];
          int km=k - iDir[2];
            // (-) faces  two cells
//                var[*bndIter] = 1.0  + m_amp * sin( m_two_pi * x[*bndIter] )
//                                      * cos( m_two_pi * y[*bndIter] );
            var(i,j,k) = 1.0  + m_amp * sin( m_two_pi * x(im,jm,km) )
                   * cos( m_two_pi * y(im,jm,km) );


            var(im,jm,km) = 1.0  + m_amp * sin( m_two_pi * x(im,jm,km) )
                                  * cos( m_two_pi * y(im,jm,km) );
          });
        } else {
          // (+) faces
        parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
          var(i,j,k) = 1.0  + m_amp * sin( m_two_pi * x(i,j,k) )
                                  * cos( m_two_pi * y(i,j,k) );

        });
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

};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::MMSshunn : BaseFunctor{

public:

  MMSshunn( std::string x_name) : BaseFunctor(), m_x_name(x_name) {}
  ~MMSshunn(){}

  void add_dep( FunctorDepList& master_dep ){

    DepBCInfo x_info; x_info.variable_name = m_x_name;
    BaseFunctor::m_dep.push_back( x_info );
    BaseFunctor::check_master_list( BaseFunctor::m_dep, master_dep );

  }

  void add_mod( std::vector<std::string>& master_mod ){}

template <typename ES, typename MS>
  void eval_bc( ExecutionObject<ES,MS>& executionObject,std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::ListOfCellsIterator& bndIter  ){

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
        parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
        const double z2 = std::cosh(m_w0 * std::exp (-m_k2 * time_d) *x(i,j,k)); // x is cc value
        const double phi = (z1-z2)/(z1 * (1.0 - m_rho0/m_rho1)-z2);
        const double rho = 1.0/(phi/m_rho1 + (1.0- phi )/m_rho0);
        var(i,j,k) = phi*rho;
       });
     }


    }
private:

  std::string m_x_name;

};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::SecondaryVariableBC : BaseFunctor {

public:

  SecondaryVariableBC( std::string sec_var_name ) : m_sec_var_name(sec_var_name){
    DepBCInfo sec_info; sec_info.variable_name = m_sec_var_name;
    BaseFunctor::m_dep.push_back( sec_info );
  }
  ~SecondaryVariableBC(){}

  void add_dep( FunctorDepList& master_dep ){

    // Now adding dependencies to the master list.
    // This checks for repeats to ensure a variable isn't added twice.
    BaseFunctor::check_master_list( BaseFunctor::m_dep, master_dep );

  }

  void add_mod( std::vector<std::string>& master_mod ){}

template <typename ES, typename MS>
  void eval_bc(ExecutionObject<ES,MS>& executionObject, std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::ListOfCellsIterator& bndIter  ){

    VariableHelper<T> var_help;
    typedef typename VariableHelper<T>::ConstType CT;
    T& var = *( tsk_info->get_uintah_field<T>(var_name));
    const IntVector iDir = patch->faceDirection( bnd->face );
    CT& sec_var = *( tsk_info->get_const_uintah_field<CT>(m_sec_var_name));

    parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
          int im=i - iDir[0];
          int jm=j - iDir[1];
          int km=k - iDir[2];
          double interp_sec_var = (sec_var(i,j,k) + sec_var(im,jm,km))/2.;
          var(i,j,k) = 2. * interp_sec_var - var(im,jm,km);
    });

  }

private:

  std::string m_sec_var_name;
  MaterialManagerP m_materialManager;

};
//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::PressureOutletBC : BaseFunctor {

public:

  PressureOutletBC( std::string vel_name, double value ):m_vel_name(vel_name), m_vel_value(value){
  }
  ~PressureOutletBC(){}

  void add_dep( FunctorDepList& master_dep ){
    // Now adding dependencies to the master list.
    // This checks for repeats to ensure a variable isn't added twice.
    //DepBCInfo vel_info; vel_info.variable_name = m_vel_name;
    //vel_info.dw = ArchesFieldContainer::LATEST; 
    //BaseFunctor::m_dep.push_back( vel_info );
    //BaseFunctor::check_master_list( BaseFunctor::m_dep, master_dep );

  }

  void add_mod( std::vector<std::string>& master_mod ){}
  

template <typename ES, typename MS>
  void eval_bc(ExecutionObject<ES,MS>& executionObject, std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::ListOfCellsIterator& bndIter  ){

    // There is an issue with register variable for old_var. This BC is being applied in VelRhoHatBC.cc 

    //typedef typename VariableHelper<T>::ConstType CT;
    //T& var = *( tsk_info->get_uintah_field<T>(var_name));

    //constCCVariable<double>& old_var =
    //tsk_info->get_const_uintah_field_add<constCCVariable<double> >(m_vel_name);

    //const double possmall = 1e-16;
    //const IntVector iDir = patch->faceDirection( bnd->face );
    //Patch::FaceType face = bnd->face;
    //BndTypeEnum my_type  = bnd->type;
    //int sign = iDir[0] + iDir[1] + iDir[2];
    //int bc_sign = 0;
    //int move_to_face_value = ( sign < 1 ) ? 1 : 0;

    //IntVector move_to_face(std::abs(iDir[0])*move_to_face_value,
    //                       std::abs(iDir[1])*move_to_face_value,
    //                       std::abs(iDir[2])*move_to_face_value);

    //if ( my_type == OUTLET ){
    //  bc_sign = 1.;
    //} else if ( my_type == PRESSURE){
    //  bc_sign = -1.;
    //}

    //sign = bc_sign * sign;

    //if ( my_type == OUTLET || my_type == PRESSURE ){
      // This applies the mostly in (pressure)/mostly out (outlet) boundary condition
    //  parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (const int i,const int j,const int k) {
    //    int i_f = i + move_to_face[0]; // cell on the face
    //    int j_f = j + move_to_face[1];
    //    int k_f = k + move_to_face[2];
  
    //    int im = i_f - iDir[0];// first interior cell
    //    int jm = j_f - iDir[1];
    //    int km = k_f - iDir[2];
  
    //    int ipp = i_f + iDir[0];// extra cell face in the last index (mostly outwardly position) 
    //    int jpp = j_f + iDir[1];
    //    int kpp = k_f + iDir[2];
  
    //    if ( sign * old_var(i_f,j_f,k_f) > possmall ){
        //if ( sign * var(i_f,j_f,k_f) > possmall ){
          // du/dx = 0
    //      var(i_f,j_f,k_f)= var(im,jm,km);
    //    } else {
          // shut off the hatted value to encourage the mostly-* condition
    //      var(i_f,j_f,k_f) = 0.0;
    //    }
  
    //    var(ipp,jpp,kpp) = var(i_f,j_f,k_f);
  
    //  });
    //}

    }

private:

  std::string m_vel_name;
  const double m_vel_value;

};


//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::VelocityBC : BaseFunctor {

public:

  VelocityBC( std::string density_name, double value ):m_density_name(density_name), m_vel_value(value){
  }
  ~VelocityBC(){}

  void add_dep( FunctorDepList& master_dep ){

    // Now adding dependencies to the master list.
    // This checks for repeats to ensure a variable isn't added twice.
    DepBCInfo den_info; den_info.variable_name = m_density_name;
    BaseFunctor::m_dep.push_back( den_info );
    BaseFunctor::check_master_list( BaseFunctor::m_dep, master_dep );

  }

  void add_mod( std::vector<std::string>& master_mod ){}

template <typename ES, typename MS>
  void eval_bc(ExecutionObject<ES,MS>& executionObject, std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::ListOfCellsIterator& bndIter  ){

    VariableHelper<T> var_help;
    //typedef typename VariableHelper<T>::ConstType CT;
    T& var = *( tsk_info->get_uintah_field<T>(var_name));
    constCCVariable<double>& rho =
      *( tsk_info->get_const_uintah_field<constCCVariable<double> >(m_density_name));

    const IntVector iDir = patch->faceDirection( bnd->face );

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

    const IntVector offset_iDir = iDir + offset;

    IntVector vDir(var_help.ioff, var_help.joff, var_help.koff);
    const double dot = vDir[0]*iDir[0] + vDir[1]*iDir[1] + vDir[2]*iDir[2];


    if ( parallel_dir ){
        //The face normal and the velocity are in parallel
        if (dot == -1) {
            //Face +
           parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
                 int im=i - iDir[0];
                 int jm=j - iDir[1];
                 int km=k - iDir[2];
                 const double interp_rho = 0.5*(rho(im,jm,km)+rho(i,j,k));
                 var(im,jm,km) = interp_rho*m_vel_value;
                 var(i,j,k) = var(im,jm,km);
            });
        } else {
            // Face -
           parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
                 const double interp_rho = 0.5*(rho[*bndIter-iDir]+rho[*bndIter]);
                  var(i,j,k) = interp_rho*m_vel_value;
                  //var[*bndIter] = var[*bndIter-iDir];
           });
       }
    } else {
      //The face normal and the velocity are tangential
        parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {
        int im=i - iDir[0];
        int jm=j - iDir[1];
        int km=k - iDir[2];

        int imo=i - offset_iDir[0];
        int jmo=j - offset_iDir[1];
        int kmo=k - offset_iDir[2];

        int ipo=i + offset[0];
        int jpo=j + offset[1];
        int kpo=k + offset[2];

        const double interp_rho =0.5*(0.5*(rho(im,jm,km)+rho(i,j,k)) +
                                      0.5*(rho(imo,jmo,kmo)+rho(ipo,jpo,kpo)));

        var(i,j,k) = 2.*interp_rho*m_vel_value - var(im,jm,km);

      });
    }
  }

private:

  std::string m_density_name;
  const double m_vel_value;

};

//--------------------------------------------------------------------------------------------------
template <typename T>
struct BCFunctors<T>::SubGridInjector : BaseFunctor {

public:

  SubGridInjector( const std::string phi_name, ProblemSpecP db )
  : m_phi_name(phi_name) {

    //value in this context is rho*phi
    db->require("value", m_rho_phi_u);

    m_use_face_area = true;
    if ( db->findBlock("area") ){
      db->require( "area", m_area );
      m_use_face_area = false;
    }

  }
  ~SubGridInjector(){}

  void add_dep( FunctorDepList& master_dep ){}

  void add_mod( std::vector<std::string>& master_mod ){

    const std::string rhs_name = m_phi_name+"_rhs";

    auto i = find(master_mod.begin(), master_mod.end(), rhs_name);

    if ( i == master_mod.end() ){
      master_mod.push_back(rhs_name);
    }

  }

template <typename ES, typename MS>
  void eval_bc(ExecutionObject<ES,MS>& executionObject, std::string var_name, const Patch* patch, ArchesTaskInfoManager* tsk_info,
                const BndSpec* bnd, Uintah::ListOfCellsIterator& bndIter  ){

    VariableHelper<T> var_help;
    T& var = tsk_info->get_uintah_field_add<T>(var_name);

    T& rhs = tsk_info->get_uintah_field_add<T>(m_phi_name+"_rhs");

    const IntVector iDir = patch->faceDirection( bnd->face );

    double area = 0.0;
    Vector DX = patch->dCell();
    if ( m_use_face_area ){
      area = std::abs(iDir[0]) * DX.y() * DX.z() +
             std::abs(iDir[1]) * DX.z() * DX.x() +
             std::abs(iDir[2]) * DX.x() * DX.y();
    } else {
      area = m_area;
    }

    parallel_for(bndIter.get_ref_to_iterator(),bndIter.size(), [&] (int i,int j,int k) {

      rhs(i,j,k) += m_rho_phi_u * area;

    });
  }

private:

  const std::string m_phi_name;
  double m_rho_phi_u;
  double m_area;
  bool m_use_face_area;

};

// <put new functors here>

//--------- END BC FUNCTORS --------------

//--------------------------------------------------------------------------------------------------
template <typename T>
void BCFunctors<T>::get_bc_dependencies( std::vector<std::string>& varnames, WBCHelper* bc_helper,
                                         FunctorDepList& dep ){

  const BndMapT& bc_info = bc_helper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    std::string facename = i_bc->second.name;

    for ( auto i_eqn = varnames.begin(); i_eqn != varnames.end(); i_eqn++ ){

      const BndCondSpec* spec = i_bc->second.find(*i_eqn);

      if ( spec == nullptr ){

        // Only throwing an error for edge BCs.
        // Interior BC's allowed to be missing
        if ( i_bc->second.edge_type == EDGE ){
          std::stringstream msg;
          msg << "Error: Cannot find a boundary condition for variable: " << *i_eqn << " on face: " << facename << std::endl;
          throw InvalidValue(msg.str(), __FILE__, __LINE__);
        }

      } else {

        std::shared_ptr<BaseFunctor> bc_fun = nullptr;
        if ( spec->bcType == CUSTOM ){
          //CUSTOM BCS (i.e., NOT Dirichlet or Neumann)
          std::string key_name = pair_face_var_names( facename, *i_eqn );
          bc_fun = m_bcFunStorage[key_name];
        }

        if ( bc_fun != nullptr ){
          bc_fun->add_dep( dep );
        }

      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
template <typename T>
void BCFunctors<T>::get_bc_modifies( std::vector<std::string>& varnames, WBCHelper* bc_helper,
                                     std::vector<std::string>& mod ){

  const BndMapT& bc_info = bc_helper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    std::string facename = i_bc->second.name;

    for ( auto i_eqn = varnames.begin(); i_eqn != varnames.end(); i_eqn++ ){

      const BndCondSpec* spec = i_bc->second.find(*i_eqn);

      if ( spec == nullptr ){

        // Only throwing an error for edge BCs.
        // Interior BC's allowed to be missing
        if ( i_bc->second.edge_type == EDGE ){
          std::stringstream msg;
          msg << "Error: Cannot find a boundary condition for variable: " << *i_eqn << " on face: " << facename << std::endl;
          throw InvalidValue(msg.str(), __FILE__, __LINE__);
        }

      } else {

        std::shared_ptr<BaseFunctor> bc_fun = nullptr;
        if ( spec->bcType == CUSTOM ){
          //CUSTOM BCS (i.e., NOT Dirichlet or Neumann)
          std::string key_name = pair_face_var_names( facename, *i_eqn );
          bc_fun = m_bcFunStorage[key_name];
        }

        if ( bc_fun != nullptr ){
          bc_fun->add_mod( mod );
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

    const bool on_this_patch = i_bc->second.has_patch(patch->getID());

    if ( on_this_patch ){

      //Get the iterator
      Uintah::ListOfCellsIterator& cell_iter = bc_helper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());

      std::string facename = i_bc->second.name;

      for ( auto i_eqn = varnames.begin(); i_eqn != varnames.end(); i_eqn++ ){

        const BndCondSpec* spec = i_bc->second.find(*i_eqn);

        if ( spec == nullptr ){

          // Only throwing an error for edge BCs.
          // Interior BC's allowed to be missing
          if ( i_bc->second.edge_type == EDGE ){
            std::stringstream msg;
            msg << "Error: Cannot find a boundary condition for variable: " <<
              *i_eqn << " on face: " << facename << std::endl;
            throw InvalidValue(msg.str(), __FILE__, __LINE__);
          }

        } else {

          std::shared_ptr<BaseFunctor> bc_fun = nullptr;
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
          if ( bc_fun != nullptr ){
            //bc_fun->eval_bc<UintahSpaces::CPU, UintahSpaces::HostSpace>( *i_eqn, patch, tsk_info, &bndSpec, cell_iter );
            ExecutionObject<int,int> place_holder;
            bc_fun->eval_bc(place_holder, *i_eqn, patch, tsk_info, &bndSpec, cell_iter );
          } else {
            std::stringstream msg;
            msg << "Error: Boundary condition implementation not found." << " \n " <<
                   " Var name:     " << spec->varName << std::endl <<
                   "  Face name:    " << facename << std::endl <<
                   "  Functor name: " << spec->functorName << std::endl;
            throw InvalidValue( msg.str(), __FILE__, __LINE__);
          }
        }
      }
    }
  }
}

} } //end namespace Uintah::BoundaryFunctor


#endif
