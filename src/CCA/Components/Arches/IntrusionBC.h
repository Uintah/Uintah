#ifndef Uintah_Components_Arches_IntrusionBC_h
#define Uintah_Components_Arches_IntrusionBC_h

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/LevelP.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/Grid/Box.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Util/DebugStream.h>
#include <CCA/Components/Arches/HandoffHelper.h>
#include <Core/Grid/Patch.h>

#include <mutex>


//============================================

/**
 * @class  IntrusionBC
 * @author Jeremy Thornock
 * @date   Sep, 2011
 *
 * @brief  Sets boundary conditions for special intrusions which
 *         can act as any type of boundary condition for specified
 *         regions of the intrusion.
 *
 */

namespace Uintah{

  // setenv SCI_DEBUG INTRUSION_DEBUG:+
  static DebugStream cout_intrusiondebug("INTRUSION_DEBUG",false);

  class VarLabel;
  class ArchesLabel;
  class MPMArchesLabel;
  class ArchesVariables;
  class ArchesConstVariables;
  class Properties;
  class TableLookup;

  class IntrusionBC {

    public:

      enum INTRUSION_TYPE { INLET, SIMPLE_WALL };

      IntrusionBC( const ArchesLabel* lab, const MPMArchesLabel* mpmlab, Properties* props,
                   TableLookup* table_lookup, int WALL );
      ~IntrusionBC();

      /** @brief Interface to input file */
      void problemSetup( const ProblemSpecP& params, const int ilvl );

      /** @brief Computes the boundary area for the non-wall portion */
      void sched_computeBCArea( SchedulerP& sched,
                                const LevelP& level,
                                const MaterialSet* matls );

      void computeBCArea( const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw );

      /** @brief finds intrusions intersecting with the local to the patch and prunes the rest */
      void prune_per_patch_intrusions( SchedulerP& sched,
                                   const LevelP& level,
                                   const MaterialSet* matls );

      /** @brief Computes the velocity if a mass flow rate is specified */
      void sched_setIntrusionVelocities( SchedulerP& sched,
                                         const LevelP& level,
                                         const MaterialSet* matls );

      void setIntrusionVelocities( const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw );

      /** @brief Computes the properties at the boundary */
      void sched_computeProperties( SchedulerP& sched,
                                    const LevelP& level,
                                    const MaterialSet* matls );

      void computeProperties( const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw );

      /** @brief Print a summary of the intrusion information **/
      void sched_printIntrusionInformation( SchedulerP& sched,
                                            const LevelP& level,
                                            const MaterialSet* matls );

      void printIntrusionInformation( const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw );

      /** @brief Sets the cell type, volume and area fractions @ boundaries */
      void sched_setCellType( SchedulerP& sched,
                              const LevelP& level,
                              const MaterialSet* matls,
                              const bool doing_restart);

      void setCellType( const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw,
                        const bool doing_restart);


      /** @brief Sets the hatted velocity boundary conditions */
      void setHattedVelocity( const Patch* p,
                              SFCXVariable<double>& u,
                              SFCYVariable<double>& v,
                              SFCZVariable<double>& w,
                              constCCVariable<double>& density );

      /** @brief Set the scalar value at the boundary interface */
      void setScalar( const int p,
                      const std::string scalar_name,
                      CCVariable<double>& scalar );

      /** @brief Adds flux contribution to the RHS **/
      void addScalarRHS( const Patch* p,
                         Vector Dx,
                         const std::string scalar_name,
                         CCVariable<double>& RHS,
                         constCCVariable<double>& density );

      /** @brief Adds flux contribution to the RHS, no density**/
      void addScalarRHS( const Patch* p,
                         Vector Dx,
                         const std::string scalar_name,
                         CCVariable<double>& RHS );

      /** @brief Sets the density in the intrusion for inlets */
      void setDensity( const Patch* patch,
                       CCVariable<double>& density );

      /** @brief Sets the temperature field to that of the intrusion temperature */
      void sched_setIntrusionT( SchedulerP& sched,
                                const LevelP& level,
                                const MaterialSet* matls );

      void setIntrusionT( const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw );

      //------------ scalars ---------------------
      //

      /** @brief A base class for scalar inlet conditions **/
      class scalarInletBase {

        public:

          enum ScalarBCType { CONSTANT, FROMFILE, TABULATED };

          scalarInletBase(){
            // helper for the intvector direction
            _dHelp.push_back( IntVector(-1,0,0) );
            _dHelp.push_back( IntVector(+1,0,0) );
            _dHelp.push_back( IntVector(0,-1,0) );
            _dHelp.push_back( IntVector(0,+1,0) );
            _dHelp.push_back( IntVector(0,0,-1) );
            _dHelp.push_back( IntVector(0,0,+1) );

            // helper for the indexing for face cells
            _faceDirHelp.push_back( IntVector(0,0,0) );
            _faceDirHelp.push_back( IntVector(+1,0,0) );
            _faceDirHelp.push_back( IntVector(0,0,0) );
            _faceDirHelp.push_back( IntVector(0,+1,0) );
            _faceDirHelp.push_back( IntVector(0,0,0) );
            _faceDirHelp.push_back( IntVector(0,0,+1) );

            // helper for referencing the right index depending on direction
            _iHelp.push_back( 0 );
            _iHelp.push_back( 0 );
            _iHelp.push_back( 1 );
            _iHelp.push_back( 1 );
            _iHelp.push_back( 2 );
            _iHelp.push_back( 2 );

            // helper for the sign on the face
            _sHelp.push_back( -1.0 );
            _sHelp.push_back( +1.0 );
            _sHelp.push_back( -1.0 );
            _sHelp.push_back( +1.0 );
            _sHelp.push_back( -1.0 );
            _sHelp.push_back( +1.0 );

            // helper for getting neighboring interior cell
            _inside.push_back( IntVector(-1,0,0) );
            _inside.push_back( IntVector( 0,0,0) );
            _inside.push_back( IntVector( 0,-1,0) );
            _inside.push_back( IntVector( 0,0,0) );
            _inside.push_back( IntVector( 0,0,-1) );
            _inside.push_back( IntVector( 0,0,0) );
          };

          virtual ~scalarInletBase(){};

          virtual void problem_setup( ProblemSpecP& db, ProblemSpecP& db_intrusion ) = 0;

          virtual void set_scalar_rhs( const int& dir,
                                       IntVector c,
                                       CCVariable<double>& RHS,
                                       const double& face_den,
                                       const double& face_vel,
                                       const std::vector<double>& area ) = 0;

          virtual double get_scalar( const IntVector c ) = 0;

          ScalarBCType get_type(){ return _type; };

          Vector get_relative_xyz(){ return Vector(0,0,0);}

        protected:

          std::vector<IntVector> _dHelp;
          std::vector<IntVector> _faceDirHelp;
          std::vector<IntVector> _inside;
          std::vector<int>       _iHelp;
          std::vector<double>    _sHelp;
          ScalarBCType           _type;


      };

      /** @brief Sets the scalar boundary value to a constant **/
      class constantScalar : public scalarInletBase {

        public:

          constantScalar(){ _type = CONSTANT; };
          ~constantScalar(){};

          void problem_setup( ProblemSpecP& db, ProblemSpecP& db_intrusion ){

            db->getWithDefault("constant",_C, 0.0);

          };

          void set_scalar_rhs( const int& dir,
                               IntVector c,
                               CCVariable<double>& RHS,
                               const double& face_den,
                               const double& face_vel,
                               const std::vector<double>& area ){

            RHS[ c ] += _sHelp[dir] * area[dir] * face_den * face_vel * _C;

          };

          inline double get_scalar( const IntVector ){

            return _C;

          };

        private:

          double _C;

      };

      /** @brief Sets the scalar boundary value to a constant **/
      class tabulatedScalar : public scalarInletBase {

        public:

          tabulatedScalar(){ _type = TABULATED; };
          ~tabulatedScalar(){};

          void problem_setup( ProblemSpecP& db, ProblemSpecP& db_intrusion ){

            db->require("depend_varname",_var_name);

          };

          void set_scalar_rhs( const int& dir,
                               IntVector c,
                               CCVariable<double>& RHS,
                               const double& face_den,
                               const double& face_vel,
                               const std::vector<double>& area ){

            RHS[ c ] += _sHelp[dir] * area[dir] * face_den * face_vel * _C;

          };

          inline double get_scalar( const IntVector ){

            return _C;

          };

          void set_scalar_constant( double C ){ _C = C; };

          std::string get_depend_var_name(){ return _var_name; };

        private:

          double _C;
          std::string _var_name;

      };

      class scalarFromInput : public scalarInletBase {

        public:

          typedef std::map<IntVector, double> CellToValuesMap;
          typedef std::map<std::string, CellToValuesMap> ScalarToBCValueMap;

          scalarFromInput(std::string label) : _label(label){
            _type = FROMFILE;
            m_handoff_helper = scinew ArchesCore::HandoffHelper();
          };
          ~scalarFromInput(){
            delete m_handoff_helper;
          };

          void problem_setup( ProblemSpecP& db, ProblemSpecP& db_intrusion ){

            std::string inputfile;
            Vector relative_xyz;
            std::string default_value;

            db->require("input_file",inputfile);
            db->require("relative_xyz",relative_xyz);
            //NOTE: Only allowing default type = DIRICHLET
            db->require("default_value",default_value);

            for ( ProblemSpecP db_flux = db_intrusion->findBlock("flux_dir"); db_flux != nullptr; db_flux = db_flux->findNextBlock("flux_dir") ){

              std::string my_dir;
              my_dir = db_flux->getNodeValue();
              if ( my_dir == "x-" || my_dir == "X-"){

                _flux_i = 0;

              } else if ( my_dir == "x+" || my_dir == "X+"){

                _flux_i = 0;

              } else if ( my_dir == "y-" || my_dir == "Y-"){

                _flux_i = 1;

              } else if ( my_dir == "y+" || my_dir == "Y+"){

                _flux_i = 1;

              } else if ( my_dir == "z-" || my_dir == "Z-"){

                _flux_i = 2;

              } else if ( my_dir == "z+" || my_dir == "Z+"){

                _flux_i = 2;

              } else {
                proc0cout << "Warning: Intrusion flux direction = " << my_dir << " not recognized.  Ignoring...\n";
              }
            }

            m_handoff_helper->readInputFile( inputfile, -1, m_handoff_information );

          };

          Vector get_relative_xyz(){ return m_handoff_information.relative_xyz;}

          void set_scalar_rhs( const int& dir,
                               IntVector c,
                               CCVariable<double>& RHS,
                               const double& face_den,
                               const double& face_vel,
                               const std::vector<double>& area ){

            IntVector c_int = c;
            c_int[_flux_i] = 0;
            CellToValuesMap::iterator iter = m_handoff_information.values.find( c_int );

            if ( iter != m_handoff_information.values.end() ){
              double scalar_value = iter->second;
              RHS[ c ] += _sHelp[dir] * area[dir] * face_den * face_vel * scalar_value;
            } else {
              //default condition - ONLY ALLOWING DIRICHLET
              RHS[ c ] += _sHelp[dir] * area[dir] * face_den * face_vel * m_handoff_information.default_value;
            }

          };

          inline double get_scalar( const IntVector c ){

            IntVector c_int = c;
            c_int[_flux_i] = 0;
            CellToValuesMap::iterator iter = _bc_values.find( c_int );
            double scalar_value = iter->second;

            return scalar_value;

          };

        private:

          std::string _label;
          std::string _filename;
          CellToValuesMap _bc_values;
          int _flux_i;
          ArchesCore::HandoffHelper* m_handoff_helper;
          ArchesCore::HandoffHelper::FFInfo m_handoff_information;

      };

      //------------- velocity -----------------------
      //

      /** @brief A base class for velocity inlet conditons **/
      class VelInletBase {

        public:

          VelInletBase(){
            // helper for the intvector direction
            _dHelp.push_back( IntVector(-1,0,0) );
            _dHelp.push_back( IntVector(+1,0,0) );
            _dHelp.push_back( IntVector(0,-1,0) );
            _dHelp.push_back( IntVector(0,+1,0) );
            _dHelp.push_back( IntVector(0,0,-1) );
            _dHelp.push_back( IntVector(0,0,+1) );

            // helper for the indexing for face cells
            _faceDirHelp.push_back( IntVector(0,0,0) );
            _faceDirHelp.push_back( IntVector(+1,0,0) );
            _faceDirHelp.push_back( IntVector(0,0,0) );
            _faceDirHelp.push_back( IntVector(0,+1,0) );
            _faceDirHelp.push_back( IntVector(0,0,0) );
            _faceDirHelp.push_back( IntVector(0,0,+1) );

            // helper for referencing the right index depending on direction
            _iHelp.push_back( 0 );
            _iHelp.push_back( 0 );
            _iHelp.push_back( 1 );
            _iHelp.push_back( 1 );
            _iHelp.push_back( 2 );
            _iHelp.push_back( 2 );

            // helper for the sign on the face
            _sHelp.push_back( -1.0 );
            _sHelp.push_back( +1.0 );
            _sHelp.push_back( -1.0 );
            _sHelp.push_back( +1.0 );
            _sHelp.push_back( -1.0 );
            _sHelp.push_back( +1.0 );

            // helper for getting neighboring interior cell
            _inside.push_back( IntVector(-1,0,0) );
            _inside.push_back( IntVector( 0,0,0) );
            _inside.push_back( IntVector( 0,-1,0) );
            _inside.push_back( IntVector( 0,0,0) );
            _inside.push_back( IntVector( 0,0,-1) );
            _inside.push_back( IntVector( 0,0,0) );
          };
          virtual ~VelInletBase(){};

          virtual void problem_setup( ProblemSpecP& db ) = 0;

          virtual void set_velocity( int dir,
                                     IntVector c,
                                     SFCXVariable<double>& u,
                                     SFCYVariable<double>& v,
                                     SFCZVariable<double>& w,
                                     constCCVariable<double>& den,
                                     double bc_density ) = 0;

          virtual Vector const get_velocity( const IntVector ) = 0;

          virtual void massflowrate_velocity( int d, const double value ) = 0;

        protected:

          std::vector<IntVector> _dHelp;
          std::vector<IntVector> _faceDirHelp;
          std::vector<IntVector> _inside;
          std::vector<int>       _iHelp;
          std::vector<double>    _sHelp;


      };

      /** @brief Flat velocity profile */
      class FlatVelProf : public VelInletBase {

        // Sets normal velocity to:
        // u = 2*rho_b*u_b/(rho_b + rho_flow);

        public:

          FlatVelProf(){};
          ~FlatVelProf(){};

          void problem_setup( ProblemSpecP& db ){


            double u;
            double v;
            double w;

            ProblemSpecP db_flat = db->findBlock("velocity");

            bool is_mass_flow_rate = false;
            std::string kind;
            db_flat->getAttribute("type",kind);

            if ( kind == "massflow" ){
              is_mass_flow_rate = true;
            }

            if ( db_flat && !is_mass_flow_rate ) {
              db_flat->getWithDefault("u",u,0.0);
              db_flat->getWithDefault("v",v,0.0);
              db_flat->getWithDefault("w",w,0.0);
            } else {
              u=0.0;
              v=0.0;
              w=0.0;
            }

            _bc_velocity[0] = u;
            _bc_velocity[1] = v;
            _bc_velocity[2] = w;

          };

          inline void set_velocity( int dir,
                                    IntVector c,
                                    SFCXVariable<double>& u,
                                    SFCYVariable<double>& v,
                                    SFCZVariable<double>& w,
                                    constCCVariable<double>& density,
                                    double bc_density ){

            if ( dir == 0 || dir == 1 ){

              u[c] = _bc_velocity[0];

            } else if ( dir == 2 || dir == 3 ){

              v[c] = _bc_velocity[1];

            } else {

              w[c] = _bc_velocity[2];

            }
          };

          const inline Vector get_velocity( const IntVector ){
            return _bc_velocity;
          }

          void massflowrate_velocity( int d, const double v ){
            _bc_velocity[d] = v;
          }

        private:

          Vector _bc_velocity;
      };

      /** @brief Velocity File from an input file **/
      class InputFileVelocity : public VelInletBase {

        public:

          InputFileVelocity(){};
          ~InputFileVelocity(){};

          typedef std::map<IntVector, double> CellToValuesMap;
          typedef std::map<std::string, CellToValuesMap> ScalarToBCValueMap;

          void problem_setup( ProblemSpecP& db ){

            ProblemSpecP db_v = db->findBlock("velocity");

            Vector relative_xyz;
            std::string default_type;
            std::string default_value;

            db_v->require("input_file",_file_reference);
            db->require("relative_xyz",relative_xyz);
            //NOTE: Only allowing default type = DIRICHLET
            db->require("default_value",default_value);

            int num_flux_dir = 0; // Only allow for ONE flux direction

            for ( ProblemSpecP db_flux = db->findBlock("flux_dir"); db_flux != nullptr; db_flux = db_flux->findNextBlock("flux_dir") ){

              std::string my_dir;
              my_dir = db_flux->getNodeValue();
              if ( my_dir == "x-" || my_dir == "X-"){

                _flux_i = 0;
                num_flux_dir += 1;

              } else if ( my_dir == "x+" || my_dir == "X+"){

                _flux_i = 0;
                num_flux_dir += 1;

              } else if ( my_dir == "y-" || my_dir == "Y-"){

                _flux_i = 1;
                num_flux_dir += 1;

              } else if ( my_dir == "y+" || my_dir == "Y+"){

                _flux_i = 1;
                num_flux_dir += 1;

              } else if ( my_dir == "z-" || my_dir == "Z-"){

                _flux_i = 2;
                num_flux_dir += 1;

              } else if ( my_dir == "z+" || my_dir == "Z+"){

                _flux_i = 2;
                num_flux_dir += 1;

              } else {
                proc0cout << "Warning: Intrusion flux direction = " << my_dir << " not recognized.  Ignoring...\n";
              }
            }

            if ( num_flux_dir == 0 || num_flux_dir > 1 ){
              throw ProblemSetupException("Error: Only one flux_dir allowed. ", __FILE__, __LINE__);
            }

            m_handoff_helper->readInputFile( _file_reference, _flux_i, m_handoff_information );

          };

          inline void set_velocity( int dir,
                                    IntVector c,
                                    SFCXVariable<double>& u,
                                    SFCYVariable<double>& v,
                                    SFCZVariable<double>& w,
                                    constCCVariable<double>& density,
                                    double bc_density ){

            IntVector c_int = c;
            c_int[_flux_i] = 0;

            if ( dir == 0 || dir == 1 ) {

              ScalarToBCValueMap::iterator u_storage = _velocity_map.find("u");
              CellToValuesMap::iterator u_iter = u_storage->second.find( c_int );

              if ( u_iter == u_storage->second.end() ){
                throw InvalidValue("Error: Can't match input file u velocity with face iterator",__FILE__,__LINE__);
              } else {
                u[c] = u_iter->second;
              }

            } else if ( dir == 2 || dir == 3 ) {

              ScalarToBCValueMap::iterator v_storage = _velocity_map.find("v");
              CellToValuesMap::iterator v_iter = v_storage->second.find( c_int );

              if ( v_iter == v_storage->second.end() ){
                throw InvalidValue("Error: Can't match input file v velocity with face iterator",__FILE__,__LINE__);
              } else {
                v[c] = v_iter->second;
              }

            } else {

              ScalarToBCValueMap::iterator w_storage = _velocity_map.find("w");
              CellToValuesMap::iterator w_iter = w_storage->second.find( c_int );

              if ( w_iter == w_storage->second.end() ){
                throw InvalidValue("Error: Can't match input file w velocity with face iterator",__FILE__,__LINE__);
              } else {
                w[c] = w_iter->second;
              }

            }

          }


          const inline Vector get_velocity( const IntVector c ){
            Vector vel_vec;

            IntVector c_int = c;
            c_int[_flux_i] = 0;
            ScalarToBCValueMap::iterator u_storage = _velocity_map.find("u");
            CellToValuesMap::iterator u_iter = u_storage->second.find( c_int );
            if ( u_iter == u_storage->second.end() ){
              throw InvalidValue("Error: Can't match input file u velocity with face iterator",__FILE__,__LINE__);
            } else {
              vel_vec[0] = u_iter->second;
            }

            ScalarToBCValueMap::iterator v_storage = _velocity_map.find("v");
            CellToValuesMap::iterator v_iter = v_storage->second.find( c_int );
            if ( v_iter == v_storage->second.end() ){
              throw InvalidValue("Error: Can't match input file v velocity with face iterator",__FILE__,__LINE__);
            } else {
              vel_vec[1] = v_iter->second;
            }

            ScalarToBCValueMap::iterator w_storage = _velocity_map.find("w");
            CellToValuesMap::iterator w_iter = w_storage->second.find( c_int );
            if ( w_iter == w_storage->second.end() ){
              throw InvalidValue("Error: Can't match input file w velocity with face iterator",__FILE__,__LINE__);
            } else {
              vel_vec[2] = w_iter->second;
            }

            return vel_vec;
          }

          void massflowrate_velocity( int d, const double v ){
              throw InvalidValue("Error: Not allowed to specify mass flow rate for intrusion + inputfile for velocity",__FILE__,__LINE__);
          }

        private:

          std::string _file_reference;
          std::map<IntVector, double> _u;
          std::map<IntVector, double> _v;
          std::map<IntVector, double> _w;
          int _flux_i;
          ArchesCore::HandoffHelper* m_handoff_helper;
          ArchesCore::HandoffHelper::FFInfo m_handoff_information;

          std::string _u_filename;
          std::string _v_filename;
          std::string _w_filename;

          ScalarToBCValueMap _velocity_map;

          //---- read the file ---
          std::map<IntVector, double>
          readInputFile( std::string file_name )
          {

            gzFile file = gzopen( file_name.c_str(), "r" );
            if ( file == nullptr ) {
              proc0cout << "Error opening file: " << file_name << " for boundary conditions. Errno: " << errno << std::endl;
              throw ProblemSetupException("Unable to open the given input file: " + file_name, __FILE__, __LINE__);
            }

            std::string variable = getString( file );
            int         num_points = getInt( file );
            std::map<IntVector, double> result;

            for ( int i = 0; i < num_points; i++ ) {
              int I = getInt( file );
              int J = getInt( file );
              int K = getInt( file );
              double v = getDouble( file );

              IntVector C(I,J,K);
              C[_flux_i] = 0;

              result.insert( std::make_pair( C, v ));

            }

            gzclose( file );
            return result;
          }

      };

      typedef std::map<int, std::vector<IntVector> > BCIterator;

      struct Boundary {

        // The name of the intrusion is the key value in the map that stores all intrusions
        INTRUSION_TYPE                type;
        std::vector<GeometryPieceP>   geometry;
        std::vector<const VarLabel*>  labels;
        std::map<std::string, double> varnames_values_map;
        std::vector<std::string>      VARIABLE_TYPE;
        // Note that directions is a vector as: [-X,+X,-Y,+Y,-Z,+Z] ~ 0 means "off"/non-zero means "on"
        std::vector<int>              directions;
        double                        mass_flow_rate;
        BCIterator                    bc_face_iterator;       //face iterator at the outflow
        BCIterator                    interior_cell_iterator; //first flow cell adjacent to outflow
        BCIterator                    bc_cell_iterator;       //first interior wall cell at outflow
        bool                          has_been_initialized;
        Vector                        velocity;
        std::string                   name;

        //Approximate thin walls:
        bool                          thin_wall; //True: then treat intersecting thin walls as a full wall cell
        double                        thin_wall_delta; //Fraction of dx,dy,or dz that is used as a threshold.

        //state space information:
        double density; // from state-space calculation
        std::map<IntVector, double> density_map;

        //geometric information:
        const VarLabel* bc_area;

        //inlet generator
        IntrusionBC::VelInletBase* velocity_inlet_generator;
        bool has_velocity_model;

        //material properties
        double temperature;

        // control the definition of the interior relative to the object
        bool inverted;

        //scalars
        std::map<std::string, scalarInletBase*> scalar_map;

        // ignore missing bc spec
        bool ignore_missing_bc;                            /// Don't throw an error when a bc spec is found.

      };

      typedef std::map<std::string, Boundary> IntrusionMap;

      inline bool in_or_out( IntVector c, GeometryPieceP piece, const Patch* patch, bool inverted ){

        bool test = false;
        if ( inverted ) {
          test = true;
        }

        Point p = patch->cellPosition( c );
        if ( piece->inside( p ) ) {
          if ( inverted ) {
            test = false;
          } else {
            test = true;
          }
        }

        return test;

      }

      inline std::vector<Boundary> get_intrusions(){
        return _intrusions;
      }

      std::vector<IntVector> _dHelp;
      std::vector<IntVector> _faceDirHelp;
      std::vector<IntVector> _inside;
      std::vector<int>       _iHelp;
      std::vector<double>    _sHelp;

    private:

      std::vector<Boundary> _intrusions;
      IntrusionMap _intrusion_map;
      Uintah::PatchSet* localPatches_{nullptr};
      const ArchesLabel* _lab;
      const MPMArchesLabel* _mpmlab;
      Properties* _props;
      TableLookup* _table_lookup;
      int _WALL;
      bool _intrusion_on;
      bool _do_energy_exchange;
      bool _mpm_energy_exchange;

      std::mutex _bc_face_iterator_lock{};
      std::mutex _interior_cell_iterator_lock{};
      std::mutex _bc_cell_iterator_lock{};
      std::mutex _iterator_initializer_lock{};

      const VarLabel* _T_label{nullptr};

      /** @brief Add a face iterator to the list of total iterators for this patch and face */
      void inline add_face_iterator( IntVector c, const Patch* patch, int dir, IntrusionBC::Boundary& intrusion ){

        _bc_face_iterator_lock.lock();
        {
          if ( patch->containsCell( c + _inside[dir] ) ) {

            int p = patch->getID();

            BCIterator::iterator iMAP = intrusion.bc_face_iterator.find( p );

            if ( iMAP == intrusion.bc_face_iterator.end() ) {

              //this is a new patch that hasn't been added yet
              std::vector<IntVector> cell_indices;
              cell_indices.push_back(c);
              intrusion.bc_face_iterator.insert(make_pair( p, cell_indices ));

            } else {

              //iterator already started for this patch
              // does this cell already exist in the list?

              bool already_present = false;
              for ( std::vector<IntVector>::iterator iVEC = iMAP->second.begin(); iVEC != iMAP->second.end(); iVEC++ ){
                if ( *iVEC == c ) {
                  already_present = true;
                }
              }

              if ( !already_present ) {
                //not in the list, insert it:
                iMAP->second.push_back(c);
              }
            }
          }
        }
        _bc_face_iterator_lock.unlock();
      }

      /** @brief Add a face iterator to the list of total iterators for this patch and face */
      void inline add_interior_iterator( IntVector c, const Patch* patch, int dir, IntrusionBC::Boundary& intrusion ){

        int p = patch->getID();

        _interior_cell_iterator_lock.lock();
        {
          BCIterator::iterator iMAP = intrusion.interior_cell_iterator.find( p );

          if ( patch->containsCell( c ) ){

            if ( iMAP == intrusion.bc_face_iterator.end() ) {

              //this is a new patch that hasn't been added yet
              std::vector<IntVector> cell_indices;
              cell_indices.push_back(c);
              intrusion.bc_face_iterator.insert(make_pair( p, cell_indices ));

            } else {

              //iterator already started for this patch
              // does this cell alread exisit in the list?
              bool already_present = false;
              for ( std::vector<IntVector>::iterator iVEC = iMAP->second.begin(); iVEC != iMAP->second.end(); iVEC++ ){
                if ( *iVEC == c ) {
                  already_present = true;
                }
              }

              if ( !already_present ) {
                //not in the list, insert it:
                iMAP->second.push_back(c);
              }
            }
          }
        }
        _interior_cell_iterator_lock.unlock();
      }

      /** @brief Add a cell iterator to the list of total iterators for this patch last solid cell at outflow */
      void inline add_bc_cell_iterator( IntVector c, const Patch* patch, int dir, IntrusionBC::Boundary& intrusion ){

        int p = patch->getID();

        _bc_cell_iterator_lock.lock();
        {
          BCIterator::iterator iMAP = intrusion.bc_cell_iterator.find( p );

          if ( patch->containsCell( c ) ){

            if ( iMAP == intrusion.bc_cell_iterator.end() ) {

              //this is a new patch that hasn't been added yet
              std::vector<IntVector> cell_indices;
              cell_indices.push_back(c);
              intrusion.bc_cell_iterator.insert(make_pair( p, cell_indices ));

            } else {

              //iterator already started for this patch
              // does this cell alread exisit in the list?
              bool already_present = false;
              for ( std::vector<IntVector>::iterator iVEC = iMAP->second.begin(); iVEC != iMAP->second.end(); iVEC++ ){
                if ( *iVEC == c ) {
                  already_present = true;
                }
              }

              if ( !already_present ) {
                //not in the list, insert it:
                iMAP->second.push_back(c);
              }
            }
          }
        }
        _bc_cell_iterator_lock.unlock();
      }

      void inline initialize_the_iterators( int p, IntrusionBC::Boundary& intrusion ){

        _iterator_initializer_lock.lock();
        {
          BCIterator::iterator iMAP = intrusion.bc_face_iterator.find( p );
          if ( iMAP == intrusion.bc_face_iterator.end() ) {

            //this is a new patch that hasn't been added yet
            std::vector<IntVector> cell_indices;
            cell_indices.clear();
            intrusion.bc_face_iterator.insert(make_pair( p, cell_indices ));

          }

          BCIterator::iterator iMAP2 = intrusion.interior_cell_iterator.find( p );
          if ( iMAP2 == intrusion.interior_cell_iterator.end() ) {

            //this is a new patch that hasn't been added yet
            std::vector<IntVector> cell_indices;
            cell_indices.clear();
            intrusion.interior_cell_iterator.insert(make_pair( p, cell_indices ));

          }

          BCIterator::iterator iMAP3 = intrusion.bc_cell_iterator.find( p );
          if ( iMAP3 == intrusion.bc_cell_iterator.end() ) {

            //this is a new patch that hasn't been added yet
            std::vector<IntVector> cell_indices;
            cell_indices.clear();
            intrusion.bc_cell_iterator.insert(make_pair( p, cell_indices ));

          }
        }
        _iterator_initializer_lock.unlock();

      }

      /** @brief Prints a list of the iterators for a given patch */
      void inline print_iterator ( int p, IntrusionBC::Boundary& intrusion ){

        BCIterator::iterator iMAP = intrusion.bc_face_iterator.find( p );
        if ( iMAP == intrusion.bc_face_iterator.end() ) {

          std::cout << "For patch = " << p << " ... no FACE iterator found for geometry " << std::endl;


        } else {

          for ( std::vector<IntVector>::iterator iVEC = iMAP->second.begin(); iVEC != iMAP->second.end(); iVEC++ ){
            IntVector v = *iVEC;
            std::cout << " For patch = " << p << " found a face interator at: " << v[0] << " " << v[1] << " " << v[2] << std::endl;
          }
        }
        BCIterator::iterator iMAP2 = intrusion.interior_cell_iterator.find( p );
        if ( iMAP2 == intrusion.interior_cell_iterator.end() ) {

          std::cout << "For patch = " << p << " ... no INTERIOR iterator found for geometry " << std::endl;


        } else {

          for ( std::vector<IntVector>::iterator iVEC = iMAP2->second.begin(); iVEC != iMAP2->second.end(); iVEC++ ){
            IntVector v = *iVEC;
            std::cout << " For patch = " << p << " found a INTERIOR interator at: " << v[0] << " " << v[1] << " " << v[2] << std::endl;
          }
        }
      }

  };
} // namespace Uintah

#endif
