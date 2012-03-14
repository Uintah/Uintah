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
#include <Core/Grid/Box.h>

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
 *         Input file interface looks like: 
 *
 *         <IntrusionBC           spec="OPTIONAL NO_DATA">
 *          <intrusion            spec="MULTIPLE NO_DATA">
 *                                attribute1="type REQUIRED STRING 'flat_inlet'"
 *                                attribute2="label REQUIRED STRING">
 *            </geom_object>                                                   <!-- geometry object associated with this intrusion --> 
 *            <boundary_direction spec="REQUIRED MULTIPLE_INTEGERS"/>          <!-- direction to apply type of BC, otherwise treated as wall --> 
 *            <variable           spec="MULTIPLE NO_DATA"                      <!-- set the boundary conditions for the relevant variables --> 
 *                                attribute1="label REQUIRED STRING"           <!-- note that state variables will be looked up from the table --> 
 *                                attribute2="value REQUIRED DOUBLE"/>         <!-- typically need to set velocities, enthalpy, indep. table vars and extra scalars --> 
 *
 *                                <!-- NOTES: --> 
 *
 *                                <!-- velocity components are specified using simple [u,v,w] labels and NOT uVelocitySPBC, etc... --> 
 *                                <!-- variable = mass_flow_rate is a specific variable that sets the velocity components based
 *                                on a specified mass flow rate. -->   
 *                                <!-- If multiple directions are entered, then the mass flow rate is divided across 
 *                                all valid face directions with non-zero velocity normal to that face. --> 
 *                                <!-- Enthalpy is computed based on independ. table variables, including heat loss --> 
 *            
 *          </intrusion>
 *         </IntrsionBC>
 *
 */

namespace Uintah{ 

  class VarLabel; 
  class ArchesLabel;
  class MPMArchesLabel; 
  class ArchesVariables; 
  class ArchesConstVariables; 
  class Properties; 

  class IntrusionBC { 

    public: 

      enum INTRUSION_TYPE { INLET, SIMPLE_WALL }; 

      IntrusionBC( const ArchesLabel* lab, const MPMArchesLabel* mpmlab, Properties* props, int WALL ); 
      ~IntrusionBC(); 

      /** @brief Interface to input file */
      void problemSetup( const ProblemSpecP& params ); 

      /** @brief Computes the boundary area for the non-wall portion */
      void sched_computeBCArea( SchedulerP& sched, 
                                const PatchSet* patches, 
                                const MaterialSet* matls ); 

      void computeBCArea( const ProcessorGroup*, 
                          const PatchSubset* patches, 
                          const MaterialSubset* matls, 
                          DataWarehouse* old_dw, 
                          DataWarehouse* new_dw ); 

      /** @brief Computes the velocity if a mass flow rate is specified */
      void sched_setIntrusionVelocities( SchedulerP& sched, 
                                const PatchSet* patches, 
                                const MaterialSet* matls ); 

      void setIntrusionVelocities( const ProcessorGroup*, 
                          const PatchSubset* patches, 
                          const MaterialSubset* matls, 
                          DataWarehouse* old_dw, 
                          DataWarehouse* new_dw ); 

      /** @brief Computes the properties at the boundary */
      void sched_computeProperties( SchedulerP& sched, 
                                const PatchSet* patches, 
                                const MaterialSet* matls ); 

      void computeProperties( const ProcessorGroup*, 
                          const PatchSubset* patches, 
                          const MaterialSubset* matls, 
                          DataWarehouse* old_dw, 
                          DataWarehouse* new_dw ); 

      /** @brief Sets the cell type, volume and area fractions @ boundaries */
      void sched_setCellType( SchedulerP& sched, 
                                const PatchSet* patches, 
                                const MaterialSet* matls ); 

      void setCellType( const ProcessorGroup*, 
                          const PatchSubset* patches, 
                          const MaterialSubset* matls, 
                          DataWarehouse* old_dw, 
                          DataWarehouse* new_dw ); 


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

      /** @brief Sets the temperature field to that of the intrusion temperature */ 
      void sched_setIntrusionT( SchedulerP& sched, 
                                const PatchSet* patches, 
                                const MaterialSet* matls );

      void setIntrusionT( const ProcessorGroup*, 
                          const PatchSubset* patches, 
                          const MaterialSubset* matls, 
                          DataWarehouse* old_dw, 
                          DataWarehouse* new_dw );

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
          }; 
          virtual ~VelInletBase(){}; 

          virtual void set_velocity( int dir, 
                                     IntVector c, 
                                     SFCXVariable<double>& u, 
                                     SFCYVariable<double>& v, 
                                     SFCZVariable<double>& w, 
                                     constCCVariable<double>& den, 
                                     double bc_density, 
                                     Vector bc_velocity ) = 0; 

        protected: 

          std::vector<IntVector> _dHelp;
          std::vector<IntVector> _faceDirHelp; 
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

          inline void set_velocity( int dir, 
                               IntVector c, 
                               SFCXVariable<double>& u, 
                               SFCYVariable<double>& v, 
                               SFCZVariable<double>& w, 
                               constCCVariable<double>& density, 
                               double bc_density, 
                               Vector bc_velocity ){ 

            double velocity = 0.0; 

            velocity = 2 * bc_density * bc_velocity[_iHelp[dir]] / ( bc_density + density[c - _faceDirHelp[dir]] ); 

            //IntVector cb = c + _faceDirHelp[dir]; 

            if ( dir == 0 || dir == 1 ){ 

              u[c] = velocity; 

            } else if ( dir == 2 || dir == 3 ){ 

              v[c] = velocity; 
              
            } else { 

              w[c] = velocity; 

            } 
          };
      }; 

      typedef std::map<int, std::vector<IntVector> > BCIterator; 

      struct Boundary { 

        // The name of the intrusion is the key value in the map that stores all intrusions 
        INTRUSION_TYPE               type; 
        std::vector<GeometryPieceP>  geometry; 
        std::vector<const VarLabel*> labels;
        std::map<std::string, double> varnames_values_map; 
        std::vector<std::string>     VARIABLE_TYPE; 
        // Note that directions is a vector as: [-X,+X,-Y,+Y,-Z,+Z] ~ 0 means off/non-zero means on
        std::vector<int>             directions; 
        Vector                       velocity; 
        double                       mass_flow_rate; 
        BCIterator                   bc_face_iterator; 

        //state space information: 
        double density; // from state-space calculation

        //geometric information: 
        const VarLabel* bc_area; 
      
        //inlet generator
        IntrusionBC::VelInletBase* velocity_inlet_generator; 

        //material properties
        double temperature; 

        bool inverted; 

      }; 

      typedef std::map<std::string, Boundary> IntrusionMap;

      inline bool in_or_out( IntVector c, GeometryPieceP piece, const Patch* patch, bool inverted ){ 

        bool test = false; 
        if ( inverted ) { 
          test = true; 
        } 
        Box geom_box = piece->getBoundingBox(); 
        Box patch_box = patch->getBox(); 
        Box intersect_box = geom_box.intersect( patch_box ); 

        if ( !(intersect_box.degenerate()) ){ 

          Point p = patch->cellPosition( c ); 
          if ( piece->inside( p ) ) { 
            if ( inverted ) { 
              test = false; 
            } else { 
              test = true; 
            } 
          } 
        }
        return test; 
      } 

      inline std::vector<Boundary> get_intrusions(){ 
        return _intrusions; 
      } 

    private: 

      std::vector<Boundary> _intrusions; 
      IntrusionMap _intrusion_map; 
      const ArchesLabel* _lab; 
      const MPMArchesLabel* _mpmlab; 
      Properties* _props;
      int _WALL; 
      bool _intrusion_on; 
      bool _do_energy_exchange; 
      bool _mpm_energy_exchange; 

      std::vector<IntVector> _dHelp;
      std::vector<IntVector> _faceDirHelp; 
      std::vector<int>       _iHelp; 
      std::vector<double>    _sHelp; 

      const VarLabel* _T_label; 


      /** @brief Add an iterator to the list of total iterators for this patch and face */ 
      void inline add_iterator( IntVector c, int p, IntrusionBC::Boundary& intrusion ){ 

        BCIterator::iterator iMAP = intrusion.bc_face_iterator.find( p );
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

      /** @brief Prints a list of the iterators for a given patch */ 
      void inline print_iterator ( int p, IntrusionBC::Boundary& intrusion ){ 

        BCIterator::iterator iMAP = intrusion.bc_face_iterator.find( p );
        if ( iMAP == intrusion.bc_face_iterator.end() ) {

          std::cout << "For patch = " << p << " ... no iterator found for geometry " << endl;


        } else { 

          for ( std::vector<IntVector>::iterator iVEC = iMAP->second.begin(); iVEC != iMAP->second.end(); iVEC++ ){ 
            IntVector v = *iVEC; 
            std::cout << " For patch = " << p << " found an interator at: " << v[0] << " " << v[1] << " " << v[2] << endl;
          } 
        } 
      } 

  }; 
} // namespace Uintah

#endif
