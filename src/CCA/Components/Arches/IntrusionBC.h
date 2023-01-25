#ifndef Uintah_Components_Arches_IntrusionBC_h
#define Uintah_Components_Arches_IntrusionBC_h

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/LevelP.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/Grid/Box.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DebugStream.h>
#include <CCA/Components/Arches/HandoffHelper.h>
#include <Core/Grid/Patch.h>

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
  static DebugStream cout_intrusiondebug("ARCHES_INTRUSION_SETUP_INFO",false);

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
      enum INLET_TYPE { FLAT, HANDOFF, MASSFLOW, TABULATED };

      IntrusionBC( const ArchesLabel* lab, const MPMArchesLabel* mpmlab, Properties* props,
                   TableLookup* table_lookup, int WALL );
      ~IntrusionBC();

      /** @brief Return true if there is a velocity type inlet **/
      bool has_intrusion_inlets(){ return _has_intrusion_inlets; };

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

      void setAlphaG( const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const bool carryForward );

      void sched_setAlphaG( SchedulerP& sched,
                            const LevelP& level,
                            const MaterialSet* matls,
                            const bool carryForward );

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

      /** @brief Compute the momentum source for the RHS of the Momentum cell for intrusion inlets. **/
      void addMomRHS( const Patch* p,
                      constSFCXVariable<double>& u,
                      constSFCYVariable<double>& v,
                      constSFCZVariable<double>& w,
                      SFCXVariable<double>& usrc,
                      SFCYVariable<double>& vsrc,
                      SFCZVariable<double>& wsrc,
                      constCCVariable<double>& density );

      /** @brief Compute the mass source **/
      void
      addMassRHS( const Patch*  patch,
                  CCVariable<double>& mass_src );

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
                       CCVariable<double>& density,
                       constCCVariable<double>& old_density );

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
                                       IntVector c_rel,
                                       CCVariable<double>& RHS,
                                       const double& face_den,
                                       const double& face_vel,
                                       const std::vector<double>& area ) = 0;

          virtual double get_scalar( const Patch* patch, const IntVector c ) = 0;

          ScalarBCType get_type(){ return _type; };

          virtual Vector get_relative_xyz(){ return Vector(0,0,0);}

          virtual bool is_flat(){ return false; }

        protected:

          std::vector<IntVector> _dHelp;
          std::vector<IntVector> _faceDirHelp;
          std::vector<IntVector> _inside;
          std::vector<int>       _iHelp;
          std::vector<double>    _sHelp;
          ScalarBCType           _type;
          int                    _flux_dir;     ///< In the case of handoff files, only this flux dir allowed.

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
                               IntVector c_rel,
                               CCVariable<double>& RHS,
                               const double& face_den,
                               const double& face_vel,
                               const std::vector<double>& area ){

            RHS[ c ] += _sHelp[dir] * area[dir] * face_den * face_vel * _C;

          };

          double get_scalar( const Patch* patch, const IntVector ){

            return _C;

          };

          bool is_flat(){ return true; }

        private:

          double _C;

      };

      /** @brief Sets the scalar boundary value to a constant **/
      class tabulatedScalar : public scalarInletBase {

        public:

          tabulatedScalar(){
            _type = TABULATED;
            _mapped_values.clear();
          };
          ~tabulatedScalar(){};

          void problem_setup( ProblemSpecP& db, ProblemSpecP& db_intrusion ){

            db->require("depend_varname",_var_name);

          };

          void set_scalar_rhs( const int& dir,
                               IntVector c,
                               IntVector c_rel,
                               CCVariable<double>& RHS,
                               const double& face_den,
                               const double& face_vel,
                               const std::vector<double>& area ){

            auto iter = _mapped_values.find(c);
            if ( iter != _mapped_values.end() ){

              RHS[ c ] += _sHelp[dir] * area[dir] * face_den * face_vel * iter->second;

            } else {
              std::stringstream msg;
              msg << "Error: tabulated value not found at cell position: (" << c[0] << ", " <<
              c[1] << ", " << c[2] << std::endl;
              throw InvalidValue(msg.str(), __FILE__, __LINE__ );
            }

          };

          double get_scalar( const Patch* patch, const IntVector c ){

            auto iter = _mapped_values.find(c);
            if ( iter != _mapped_values.end() ){
              return iter->second;
            } else {
              std::stringstream msg;
              msg << "Error: tabulated value not found at cell position: (" << c[0] << ", " <<
              c[1] << ", " << c[2] << std::endl;
              throw InvalidValue(msg.str(), __FILE__, __LINE__ );
            }

          };

          void set_scalar_constant( IntVector c, double value ){

            _mapped_values.insert( std::make_pair(c, value) );

          };

          std::string get_depend_var_name(){ return _var_name; };

        private:

          std::string _var_name;
          std::map<IntVector, double> _mapped_values;

      };

      class scalarFromInput : public scalarInletBase {

        public:

          typedef std::map<IntVector, double> CellToValuesMap;

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

            db->require("input_file",inputfile);
            db->require("relative_xyz",relative_xyz);

            int num_flux_dir = 0; // Only allow for ONE flux direction

            for ( ProblemSpecP db_flux = db_intrusion->findBlock("flux_dir"); db_flux != nullptr; db_flux = db_flux->findNextBlock("flux_dir") ){

              std::string my_dir;
              my_dir = db_flux->getNodeValue();
              if ( my_dir == "x-" || my_dir == "X-"){

                _flux_i = 0;
                num_flux_dir += 1;
                _zeroed_index = 0;

              } else if ( my_dir == "x+" || my_dir == "X+"){

                _flux_i = 0;
                num_flux_dir += 1;
                _zeroed_index = 0;

              } else if ( my_dir == "y-" || my_dir == "Y-"){

                _flux_i = 1;
                num_flux_dir += 1;
                _zeroed_index = 1;

              } else if ( my_dir == "y+" || my_dir == "Y+"){

                _flux_i = 1;
                num_flux_dir += 1;
                _zeroed_index = 1;

              } else if ( my_dir == "z-" || my_dir == "Z-"){

                _flux_i = 2;
                num_flux_dir += 1;
                _zeroed_index = 2;

              } else if ( my_dir == "z+" || my_dir == "Z+"){

                _flux_i = 2;
                num_flux_dir += 1;
                _zeroed_index = 2;

              } else {
                proc0cout << "Warning: Intrusion flux direction = " << my_dir << " not recognized.  Ignoring...\n";
              }
            }

            if ( num_flux_dir == 0 || num_flux_dir > 1 ){
              throw ProblemSetupException("Error: Only one flux_dir allowed for handoff files. ", __FILE__, __LINE__);
            }

            m_handoff_helper->readInputFile( inputfile, -1, m_handoff_information );

            m_handoff_information.relative_xyz = relative_xyz;

          };

          Vector get_relative_xyz(){
            return m_handoff_information.relative_xyz;}

          void set_scalar_rhs( const int& dir,
                               IntVector c,
                               IntVector c_rel,
                               CCVariable<double>& RHS,
                               const double& face_den,
                               const double& face_vel,
                               const std::vector<double>& area ){

            c_rel[_flux_i] = 0;
            CellToValuesMap::iterator iter = m_handoff_information.values.find( c_rel );

            if ( iter != m_handoff_information.values.end() ){
              double scalar_value = iter->second;
              RHS[ c ] += _sHelp[dir] * area[dir] * face_den * face_vel * scalar_value;
            } else {
              std::stringstream msg;
              msg << "Error: scalar not found in handoff file with relative position = " << c_rel[0] << ", " <<
              c_rel[1] << ", " << c_rel[2] << ")." << std::endl <<
              "Actual cell position is: " << c[0] << ", " <<
              c[1] << ", " << c[2] << ")." << std::endl <<
              "Check your relative_xyz spec in the input file OR check to see if you have a value at this position in your handoff file." << std::endl;
              throw InvalidValue(msg.str(), __FILE__, __LINE__);
            }

          };

          double get_scalar( const Patch* patch, const IntVector c ){

            Vector relative_xyz = this->get_relative_xyz();
            Point xyz(relative_xyz[0], relative_xyz[1], relative_xyz[2]);
            IntVector rel_ijk = patch->getLevel()->getCellIndex( xyz );
            IntVector c_rel = c - rel_ijk;

            c_rel[_zeroed_index] = 0;

            CellToValuesMap::iterator iter = m_handoff_information.values.find( c_rel );

            if ( iter != m_handoff_information.values.end() ){
              return iter->second;
            } else {
              std::stringstream msg;
              msg << "Error: scalar not found in handoff file with relative position = " << c_rel[0] << ", " <<
              c_rel[1] << ", " << c_rel[2] << ")." << std::endl <<
              "Actual cell position is: " << c[0] << ", " <<
              c[1] << ", " << c[2] << ")." << std::endl <<
              "Check your relative_xyz spec in the input file OR check to see if you have a value at this position in your handoff file." << std::endl;
              throw InvalidValue(msg.str(), __FILE__, __LINE__);
            }
          };

        private:

          std::string _label;
          std::string _filename;
          CellToValuesMap _bc_values;
          int _flux_i;
          int _zeroed_index;
          ArchesCore::HandoffHelper* m_handoff_helper;
          ArchesCore::HandoffHelper::FFInfo m_handoff_information;

      };

      //------------- velocity -----------------------
      //
      void
      getVelocityCondition( const Patch* patch, const IntVector ijk,
                            bool& found_value, Vector& velocity );

      /** @brief return the max velocity across all intrusions (inlets etc..) **/
      Vector getMaxVelocity();

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

          virtual void set_velocity( const Patch* patch,
                                     std::map<int, std::vector<IntVector> >::iterator iBC,
                                     const std::vector<int>& directions,
                                     SFCXVariable<double>& u,
                                     SFCYVariable<double>& v,
                                     SFCZVariable<double>& w,
                                     bool& set_nonnormal_values ) = 0;

          virtual Vector get_velocity( const IntVector, const Patch* patch ) = 0;

          virtual void get_velocity( const IntVector, const Patch* patch,
                                     bool& found_value, Vector& velocity ) = 0;

          virtual Vector get_max_velocity() = 0;

          virtual void massflowrate_velocity( int d, const double value ) = 0;

        protected:

          typedef std::map<int, std::vector<IntVector> > BCIterator;

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
              Vector vel;
              db_flat->require("flat_velocity", vel);
              u = vel[0];
              v = vel[1];
              w = vel[2];
            } else {
              u=0.0;
              v=0.0;
              w=0.0;
            }

            _bc_velocity[0] = u;
            _bc_velocity[1] = v;
            _bc_velocity[2] = w;

          };

          inline void set_velocity( const Patch* patch,
                                    BCIterator::iterator iBC_iter,
                                    const std::vector<int>& directions,
                                    SFCXVariable<double>& u,
                                    SFCYVariable<double>& v,
                                    SFCZVariable<double>& w,
                                    bool& set_nonnormal_values ){

            for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin();
                                                   i != iBC_iter->second.end();
                                                   i++){
              IntVector c = *i;
              for ( int idir = 0; idir < 6; idir++ ){
                if ( directions[idir] != 0 ){
                   if ( idir == 0 || idir == 1 ){
                     u[c] = _bc_velocity[0];
                   } else if ( idir == 2 || idir == 3 ){
                     v[c] = _bc_velocity[1];
                   } else {
                     w[c] = _bc_velocity[2];
                   }
                }
              }
            }
          }

          inline Vector get_velocity( const IntVector, const Patch* patch ){
            return _bc_velocity;
          }

          void get_velocity( const IntVector ijk, const Patch* patch,
                             bool& found_value, Vector& velocity ){

            found_value = true;
            velocity = _bc_velocity;

          }

          Vector get_max_velocity(){
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

          void problem_setup( ProblemSpecP& db ){

            ProblemSpecP db_v = db->findBlock("velocity");

            db_v->require("input_file", _file_reference);
            db_v->require("relative_xyz", m_relative_xyz);

            int num_flux_dir = 0; // Only allow for ONE flux direction

            for ( ProblemSpecP db_flux = db->findBlock("flux_dir"); db_flux != nullptr; db_flux = db_flux->findNextBlock("flux_dir") ){

              std::string my_dir;
              my_dir = db_flux->getNodeValue();
              if ( my_dir == "x-" || my_dir == "X-"){

                _flux_i = 0;
                num_flux_dir += 1;
                _zeroed_index = 0;

              } else if ( my_dir == "x+" || my_dir == "X+"){

                _flux_i = 1;
                num_flux_dir += 1;
                _zeroed_index = 0;

              } else if ( my_dir == "y-" || my_dir == "Y-"){

                _flux_i = 2;
                num_flux_dir += 1;
                _zeroed_index = 1;

              } else if ( my_dir == "y+" || my_dir == "Y+"){

                _flux_i = 3;
                num_flux_dir += 1;
                _zeroed_index = 1;

              } else if ( my_dir == "z-" || my_dir == "Z-"){

                _flux_i = 4;
                num_flux_dir += 1;
                _zeroed_index = 2;

              } else if ( my_dir == "z+" || my_dir == "Z+"){

                _flux_i = 5;
                num_flux_dir += 1;
                _zeroed_index = 2;

              } else {
                proc0cout << "Warning: Intrusion flux direction = " << my_dir << " not recognized.  Ignoring...\n";
              }
            }

            if ( num_flux_dir == 0 || num_flux_dir > 1 ){
              throw ProblemSetupException("Error: Only one flux_dir allowed for handoff files. ", __FILE__, __LINE__);
            }

            m_handoff_helper->readInputFile( _file_reference, m_handoff_information );

            std::map<IntVector, Vector> new_map;
            new_map.clear();

            // Now go through and zero out the zeroed index:
            for (auto iter = m_handoff_information.vec_values.begin();
                      iter != m_handoff_information.vec_values.end(); iter++ ){

             IntVector c = iter->first;
             Vector value = iter->second;

             c[_zeroed_index] = 0;

             new_map.insert(std::make_pair(c,value));

            }

            m_handoff_information.vec_values = new_map;

            double mag = 0.0;
            Vector max_vel(0,0,0);
            for ( auto i = m_handoff_information.vec_values.begin(); i != m_handoff_information.vec_values.end(); i++ ){
              double check_mag = std::sqrt( (i->second)[0]*(i->second)[0] +
                                            (i->second)[1]*(i->second)[1] +
                                            (i->second)[2]*(i->second)[2]);
              if ( check_mag > mag ){
                max_vel = i->second;
              }
            }

            m_max_vel = max_vel;

          };

          inline void set_velocity( const Patch* patch,
                                    BCIterator::iterator iBC_iter,
                                    const std::vector<int>& directions,
                                    SFCXVariable<double>& u,
                                    SFCYVariable<double>& v,
                                    SFCZVariable<double>& w,
                                    bool& set_nonnormal_values ){

            Point xyz(m_relative_xyz[0], m_relative_xyz[1], m_relative_xyz[2]);
            IntVector rel_ijk = patch->getLevel()->getCellIndex( xyz );

            for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin();
                                                   i != iBC_iter->second.end();
                                                   i++){

              IntVector c = *i;
              IntVector c_rel = *i - rel_ijk; //note that this is the relative index position
              c_rel[_zeroed_index] = 0;
              auto iter = m_handoff_information.vec_values.find(c_rel);
              if ( iter != m_handoff_information.vec_values.end() ){
                Vector bc_value = m_handoff_information.vec_values[c_rel];
                if ( _flux_i == 0  || _flux_i == 1 ){
                  //-x, +x
                  u[c] = bc_value[0];
                } else if ( _flux_i == 2 || _flux_i == 3 ){
                  //-y, +y
                  v[c] = bc_value[1];
                } else if ( _flux_i == 4 || _flux_i == 5){
                  //-z, +z
                  w[c] = bc_value[2];
                }
              }
            }
          }

          void get_velocity( const IntVector ijk, const Patch* patch,
                             bool& found_value, Vector& velocity ){

            Point xyz(m_relative_xyz[0], m_relative_xyz[1], m_relative_xyz[2]);
            IntVector rel_ijk = patch->getLevel()->getCellIndex( xyz );

            IntVector c_rel = ijk - rel_ijk; //note that this is the relative index position

            c_rel[_zeroed_index] = 0;

            auto iter = m_handoff_information.vec_values.find(c_rel);

            if ( iter != m_handoff_information.vec_values.end() ){
              velocity = m_handoff_information.vec_values[c_rel];
              found_value = true;
            } else {
              found_value = false;
            }
          }

          inline Vector get_velocity( const IntVector c, const Patch* patch ){

            Point xyz(m_relative_xyz[0], m_relative_xyz[1], m_relative_xyz[2]);
            IntVector rel_ijk = patch->getLevel()->getCellIndex( xyz );

            IntVector c_rel = c - rel_ijk; //note that this is the relative index position

            c_rel[_zeroed_index] = 0;

            auto iter = m_handoff_information.vec_values.find(c_rel);

            if ( iter != m_handoff_information.vec_values.end() ){
              return m_handoff_information.vec_values[c_rel];
            } else {
              std::stringstream msg;
              msg << "Error: Cannot locate handoff information for boundary intrusion for (relative) cell: (" <<
                c_rel[0] << ", " << c_rel[1] << ", " << c_rel[2] << ")." << std::endl <<
                "The intrusion cell boundary cell (unaltered) is: (" << c[0] << ", " << c[1] << ", " << c[2] << ")." << std::endl
                << " This was using a cell modifier of " << rel_ijk[0]<< " " << rel_ijk[1] <<  " "<< rel_ijk[2] <<  std::endl;
              throw InvalidValue(msg.str(), __FILE__, __LINE__ );
            }
          }

          Vector get_max_velocity(){

            return m_max_vel;

          }

          void massflowrate_velocity( int d, const double v ){
              throw InvalidValue("Error: Not allowed to specify mass flow rate for intrusion + inputfile for velocity",__FILE__,__LINE__);
          }

        private:

          std::string _file_reference;

          int _flux_i;
          int _zeroed_index{-1};

          ArchesCore::HandoffHelper* m_handoff_helper;
          ArchesCore::HandoffHelper::FFInfo m_handoff_information;

          Vector m_relative_xyz;
          Vector m_max_vel;
      };

      typedef std::map<int, std::vector<IntVector> > BCIterator;

      struct Boundary {

        // The name of the intrusion is the key value in the map that stores all intrusions
        INTRUSION_TYPE                type;
        INLET_TYPE                    velocity_inlet_type;
        INLET_TYPE                    scalar_inlet_type;
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
        // This area is for inlet's only - used to compute velocity for massflow inlets
        const VarLabel* inlet_bc_area;
        const VarLabel* wetted_surface_area;

        // The user defined value for the actual physical surface area
        double physical_area;

        // There are two options for modeling the surface area of a cell face:
        // Option A) The local area on a cell face is modeled as:
        // dA = dx_i * dx_j * physical_area / wetted_surface_area
        // The physical area is specified in the input file.
        // Option B) The local area on a cell face is modeled as:
        // dA = dx_i * dx_j * alpha_g
        // The constant alpha_g is specified in the input file.
        // Clearly alpha_g = physical_area / wetted_surface_area.
        // In one case, the physical area may be well known (Option A) while in the other,
        // the physical area may be hard to determine.
        // The value of alpha_g will be a CC field which one can visualize and is used in the
        // heat transfer model and any other place that might be relevant to a surface area.
        // The CC variable will apply to all exposed faces of the intrusion.
        double alpha_g;

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
      bool _has_intrusion_inlets{false};
      IntrusionMap _intrusion_map;
      Uintah::PatchSet* localPatches_{nullptr};
      const ArchesLabel* _lab;
      const MPMArchesLabel* _mpmlab;
      const VarLabel* m_alpha_geom_label;                  ///< Geometry modification factor
      Properties* _props;
      TableLookup* _table_lookup;
      int _WALL;
      bool _intrusion_on;
      bool _do_energy_exchange;
      bool _mpm_energy_exchange;

      Uintah::MasterLock _bc_face_iterator_lock{};
      Uintah::MasterLock _interior_cell_iterator_lock{};
      Uintah::MasterLock _bc_cell_iterator_lock{};
      Uintah::MasterLock _iterator_initializer_lock{};

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
