#ifndef Uintah_Component_Arches_EfficiencyCalculator_h
#define Uintah_Component_Arches_EfficiencyCalculator_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <Core/Grid/SimulationState.h>
/** 
 *
 * @class  Scalar Efficiency Calculator
 * @author Jeremy Thornock
 * @date   March, 2012
 *
 * @brief Computes various balances on scalars depending on which calculators are active. 
 *
 * @details 
 * This class offers the user the ability to use or create any type of global balance calculation.  
 * The typical process is: 
 * 1) Compute the reduction variables which consists of things like fluxes over all boundaries of interest
 * 2) Compute the actual efficiency. 
 * The input file interface will look like this: 
 *
 * <ARCHES>
 *   <efficiency_calculator>
 *     <calculator label="some_calculator" type="some_type">
 *       <...>
 *     </calculator>
 *    </efficiency_calculator>
 *    ....
 * </ARCHES>
 *
 * where <label> is a unique name, <type> is the type of calculator (see derived Calculator classes) and <...> are the options for 
 * the specific calculator.  
 *
 */


namespace Uintah { 

  class EfficiencyCalculator{ 

    public: 

      EfficiencyCalculator( const BoundaryCondition* bcs, ArchesLabel* lab ):_bcs(bcs), _a_labs(lab) {
      };
      ~EfficiencyCalculator(){

        // delete all calculators
        for ( LOC::iterator i = _my_calculators.begin(); i != _my_calculators.end(); i++ ){
          delete i->second; 
        }

      };

      bool problemSetup( const ProblemSpecP& db ){ 

        ProblemSpecP params = db; 

        if ( params->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("EfficiencyCalculator") ){ 

          ProblemSpecP ec_db = params->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("EfficiencyCalculator"); 

          for ( ProblemSpecP calc_db = ec_db->findBlock("calculator"); calc_db != 0; calc_db = calc_db->findNextBlock("calculator") ){ 

            std::string type; 
            std::string name; 
            calc_db->getAttribute("type",type); 
            calc_db->getAttribute("label",name); 
            bool check;

            if ( type == "combustion_efficiency" ){ 

              Calculator* calculator = scinew CombustionEfficiency( name,  _bcs, _a_labs ); 
              check = calculator->problemSetup( calc_db ); 

              if ( !check ){ 
                throw InvalidValue("Error: Trouble setting up combustion efficiency calculator.",__FILE__, __LINE__); 
              } 

              _my_calculators.insert(make_pair(name,calculator)); 

            } else if ( type == "mass_balance" ) { 

              Calculator* calculator = scinew MassBalance( name, _bcs, _a_labs );
              check = calculator->problemSetup( calc_db ); 

              if ( !check ){ 
                throw InvalidValue("Error: Trouble setting up mass balance calculator.",__FILE__, __LINE__); 
              } 

              _my_calculators.insert(make_pair(name,calculator)); 

            } else { 

              throw InvalidValue("Error: Efficiency calculator not recognized.",__FILE__, __LINE__); 

            } 
          } 

          return true; 

        } 

        return false; 

      }; 

      /** @brief Scheduler for computing the various efficiencies */
      void sched_computeAllScalarEfficiencies( const LevelP& level, 
                                               SchedulerP& sched ) 
      { 

        // loop through all calculators
        for ( LOC::iterator i = _my_calculators.begin(); i != _my_calculators.end(); i++ ){ 

          i->second->sched_computeReductionVars( level, sched ); 

          i->second->sched_computeEfficiency( level, sched ); 

        } 

      };

    private: 

      class Calculator{ 

        /** @class  Base Class for Efficiency calculators
         *  @author Jeremy Thornock
         *  @date   March, 2012
         *
         * @brief Serves as the base class for efficiency calculators
         *
         */

        public: 
          Calculator( std::string id, ArchesLabel* a_labs ) : _id(id), _a_labs(a_labs){};

          virtual ~Calculator(){};

          virtual bool problemSetup( const ProblemSpecP& db )=0; 

          /** @brief Should compute any summation over boundaries */ 
          virtual void sched_computeReductionVars( const LevelP& level, 
                                                   SchedulerP& sched )=0;

          /** @brief Should actually compute the efficiency */            
          virtual void sched_computeEfficiency(  const LevelP& level, 
                                                 SchedulerP& sched )=0;

        protected: 
          std::string _id; 
          ArchesLabel* _a_labs; 

          //.....

      }; 

      //_______________MASS BALANCE_____________________
      class MassBalance : public Calculator { 

        /** @class  MassBalance
         *  @author Jeremy Thornock
         *  @date   Aug 2013
         *
         *  @brief Computes a species mass balance
         *
         *  @details 
         * Computes
         *
         * In - Out + Accum = S
         *
         * where 
         *
         *  In  = mass flow in through inlets and pressure BC, \int_{IN} (\rho u \phi) \cdot dA 
         *  Out = mass flow out through outlets and pressure BC, \int_{OUT} (\rho u \phi) \cdot dA  
         *  Accum = accumulation, \frac{\partial \rho \phi}{\partial t} 
         *  S = source (not explicitly computed) 
         *
         *  Note that is S may be interpreted as the residual in cases where no source actually 
         *  exisits in the domain. 
         *
         *
         */

        public: 

          MassBalance(std::string id, const BoundaryCondition* bcs, ArchesLabel* a_lab) 
            : Calculator(id, a_lab), _bcs(bcs) {

            proc0cout << " Instantiating a calculator named: " << _id << " of type: mass_balance" << std::endl;

            _IN_label    = VarLabel::create( _id+"_in",    sum_vartype::getTypeDescription() ); 
            _OUT_label   = VarLabel::create( _id+"_out",   sum_vartype::getTypeDescription() ); 
            _ACCUM_label = VarLabel::create( _id+"_accum", sum_vartype::getTypeDescription() ); 
            _SOURCE_label = VarLabel::create( _id+"_source", sum_vartype::getTypeDescription() );
            _residual_label = VarLabel::create(  _id, sum_vartype::getTypeDescription() ); 
            _no_species = false; 
            _no_source  = false; 
          
          }; 

          ~MassBalance(){
          
            VarLabel::destroy( _IN_label ); 
            VarLabel::destroy( _OUT_label ); 
            VarLabel::destroy( _ACCUM_label ); 
            VarLabel::destroy( _SOURCE_label ); 
            VarLabel::destroy( _residual_label );

          }; 

          bool problemSetup( const ProblemSpecP& db ){

            ProblemSpecP params = db; 
        
            std::string species_label;
            std::string source_label; 

            if ( db->findBlock("scalar") ){

              db->findBlock("scalar")->getAttribute("label",species_label); 
              _phi_label = VarLabel::find( species_label );

              if ( _phi_label == 0 ){ 
                throw InvalidValue("Error: Cannot find phi label for mass balance calculator.",__FILE__, __LINE__); 
              } 

              if ( db->findBlock("one_minus_scalar") ){
                _A = 1;
                _C = 1; 
              } else { 
                _A = -1; 
                _C = 0;
              }

            } else { 
              _no_species = true; 
            }

            if ( db->findBlock("source") ){ 

              db->findBlock("source")->getAttribute("label",source_label); 
              _S_label = VarLabel::find( source_label ); 

              if ( _S_label == 0 ){ 
                throw InvalidValue("Error: Cannot find source label for mass balance calculator.",__FILE__, __LINE__); 
              } 

            } else { 
              _no_source = true; 
            } 

            return true; 
          
          }; 

          /** @brief Should compute any summation over boundaries */ 
          void sched_computeReductionVars( const LevelP& level, 
                                           SchedulerP& sched ){
        
            const std::string name =  "MassBalance::computeReductionVars";
            Task* tsk = scinew Task( name, this, 
                &MassBalance::computeReductionVars); 

            tsk->computes( _IN_label ); 
            tsk->computes( _OUT_label ); 
            tsk->computes( _ACCUM_label ); 
            tsk->computes( _SOURCE_label ); 

            tsk->requires( Task::NewDW, _a_labs->d_densityCPLabel, Ghost::None, 0 ); 
            tsk->requires( Task::OldDW, _a_labs->d_densityCPLabel, Ghost::None, 0 ); 
            tsk->requires( Task::NewDW, _a_labs->d_uVelocitySPBCLabel, Ghost::None, 0 ); 
            tsk->requires( Task::NewDW, _a_labs->d_vVelocitySPBCLabel, Ghost::None, 0 ); 
            tsk->requires( Task::NewDW, _a_labs->d_wVelocitySPBCLabel, Ghost::None, 0 ); 
            tsk->requires( Task::OldDW, _a_labs->d_sharedState->get_delt_label(), Ghost::None, 0);
            tsk->requires( Task::NewDW, _a_labs->d_cellTypeLabel, Ghost::None, 0 );

            if ( !_no_species ){
              tsk->requires( Task::NewDW, _phi_label, Ghost::None, 0 ); 
              tsk->requires( Task::OldDW, _phi_label, Ghost::None, 0 ); 
            }

            if ( !_no_source ){ 
              tsk->requires( Task::NewDW, _S_label, Ghost::None, 0 ); 
            } 

            sched->addTask( tsk, level->eachPatch(), _a_labs->d_sharedState->allArchesMaterials() ); 
          
          };

          void computeReductionVars( const ProcessorGroup* pc, 
                                     const PatchSubset* patches, 
                                     const MaterialSubset* matls, 
                                     DataWarehouse* old_dw, 
                                     DataWarehouse* new_dw )
          {
            for (int p = 0; p < patches->size(); p++) {

              const Patch* patch = patches->get(p);
              const Level* level = patch->getLevel(); 
              const int ilvl = level->getID(); 
              int archIndex = 0; // only one arches material
              int indx = _a_labs->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

              constSFCXVariable<double> u; 
              constSFCYVariable<double> v; 
              constSFCZVariable<double> w; 
              constCCVariable<double> rho; 
              constCCVariable<double> old_rho; 
              constCCVariable<double> phi; 
              constCCVariable<double> old_phi; 
              constCCVariable<double> source; 
              constCCVariable<int> cell_type; 

              new_dw->get( u, _a_labs->d_uVelocitySPBCLabel, indx, patch, Ghost::None, 0 ); 
              new_dw->get( v, _a_labs->d_vVelocitySPBCLabel, indx, patch, Ghost::None, 0 ); 
              new_dw->get( w, _a_labs->d_wVelocitySPBCLabel, indx, patch, Ghost::None, 0 ); 
              new_dw->get( rho, _a_labs->d_densityCPLabel, indx, patch, Ghost::None, 0 ); 
              new_dw->get( cell_type, _a_labs->d_cellTypeLabel, indx, patch, Ghost::None, 0 );
              old_dw->get( old_rho, _a_labs->d_densityCPLabel, indx, patch, Ghost::None, 0 ); 

              delt_vartype DT;
              old_dw->get(DT, _a_labs->d_sharedState->get_delt_label());
              double dt = DT; 

              if ( !_no_species ){
                new_dw->get( phi, _phi_label, indx, patch, Ghost::None, 0 ); 
                old_dw->get( old_phi, _phi_label, indx, patch, Ghost::None, 0 ); 
              }

              if ( !_no_source ){ 
                new_dw->get( source, _S_label, indx, patch, Ghost::None, 0 ); 
              }

              double sum_in    = 0.0;
              double sum_out   = 0.0; 
              double sum_accum = 0.0; 
              double sum_source = 0.0; 
              Vector Dx = patch->dCell(); 

              double vol = Dx.x()* Dx.y()* Dx.z(); 

              if ( _no_species ){
                for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ) { 

                  IntVector c = *iter; 

                  if ( cell_type[c] == -1 )
                    sum_accum += (rho[c] - old_rho[c])/dt * vol; 

                  if ( !_no_source ){ 
                    sum_source += source[c] * vol;
                  }

                }
              } else { 

                for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ) { 

                  IntVector c = *iter; 

                  if ( cell_type[c] == -1 ) 
                    sum_accum += (rho[c] * (_C-_A*phi[c]) - old_rho[c] * (_C-_A*old_phi[c])) / dt * vol;  

                  if ( !_no_source ){ 
                    sum_source += source[c] * vol;
                  }

                }
              }

              for ( auto bc_iter = _bcs->d_bc_information[ilvl]->begin(); 
                    bc_iter != _bcs->d_bc_information[ilvl]->end(); bc_iter++){

                if ( bc_iter->second.type != BoundaryCondition::WALL ){ 

                  std::vector<Patch::FaceType>::const_iterator bf_iter;
                  std::vector<Patch::FaceType> bf;
                  patch->getBoundaryFaces(bf);

                  for (bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++){

                    //get the face
                    Patch::FaceType face = *bf_iter;
                    IntVector insideCellDir = patch->faceDirection(face); 

                    //get the number of children
                    int numChildren = patch->getBCDataArray(face)->getNumberChildren(indx); //assumed one material

                    for (int child = 0; child < numChildren; child++){

                      double bc_value = -9; 
                      Vector bc_v_value(0,0,0); 
                      std::string bc_s_value = "NA";

                      Iterator bound_ptr;
                      std::string bc_kind = "NotSet";
                      std::string face_name;
                      getBCKind( patch, face, child, bc_iter->second.name, indx, bc_kind, face_name ); 

                      bool foundIterator = "false"; 
                      if ( bc_kind == "Tabulated" || bc_kind == "FromFile" ){ 
                        foundIterator = 
                          getIteratorBCValue<std::string>( patch, face, child, bc_iter->second.name, indx, bc_s_value, bound_ptr ); 
                      } else if ( bc_kind == "VelocityInlet" ){
                        foundIterator = 
                          getIteratorBCValue<Vector>( patch, face, child, bc_iter->second.name, indx, bc_v_value, bound_ptr );
                      } else { 
                        foundIterator = 
                          getIteratorBCValue<double>( patch, face, child, bc_iter->second.name, indx, bc_value, bound_ptr ); 
                      } 

                      if ( foundIterator ) {

                        switch(face) { 

                          case Patch::xminus: 
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 

                              if ( cell_type[ c - insideCellDir] == -1 ){
                                const double A     = Dx.y()*Dx.z();     

                                int norm = 0; 

                                const double rho_u_A_in  = get_minus_in_flux( u, rho, c, insideCellDir, norm ) * A;  
                                const double rho_u_A_out = get_minus_out_flux( u, rho, c, insideCellDir, norm ) * A;  

                                if ( !_no_species ){

                                  sum_in  += rho_u_A_in * (_C-_A*phi[c]); 
                                  sum_out += rho_u_A_out* (_C-_A*phi[c]);  

                                } else {

                                  sum_in += rho_u_A_in; 
                                  sum_out += rho_u_A_out; 

                                }
                              }

                            }
                            break; 
                          case Patch::xplus:
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              if ( cell_type[ c - insideCellDir] == -1 ){
                                const double A     = Dx.y()*Dx.z();     

                                int norm = 0; 

                                const double rho_u_A_in  = get_plus_in_flux( u, rho, c, insideCellDir, norm ) * A;  
                                const double rho_u_A_out = get_plus_out_flux( u, rho, c, insideCellDir, norm ) * A;  

                                if ( !_no_species ){

                                  sum_in += rho_u_A_in * (_C-_A*phi[c]); 
                                  sum_out += rho_u_A_out* (_C-_A*phi[c]);  

                                } else {

                                  sum_in += rho_u_A_in; 
                                  sum_out += rho_u_A_out; 

                                }
                              }

                            }
                            break; 
                          case Patch::yminus: 
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              if ( cell_type[ c - insideCellDir] == -1 ){
                                const double A     = Dx.x()*Dx.z();     

                                int norm = 1; 

                                const double rho_u_A_in  = get_minus_in_flux( v, rho, c, insideCellDir, norm ) * A;  
                                const double rho_u_A_out = get_minus_out_flux( v, rho, c, insideCellDir, norm ) * A;  

                                if ( !_no_species ){

                                  sum_in += rho_u_A_in * (_C-_A*phi[c]); 
                                  sum_out += rho_u_A_out* (_C-_A*phi[c]);  

                                } else {

                                  sum_in += rho_u_A_in; 
                                  sum_out += rho_u_A_out; 

                                }
                              }

                            }
                            break; 
                          case Patch::yplus:
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              if ( cell_type[ c - insideCellDir] == -1 ){
                                const double A     = Dx.x()*Dx.z();     

                                int norm = 1; 

                                const double rho_u_A_in  = get_plus_in_flux( v, rho, c, insideCellDir, norm ) * A;  
                                const double rho_u_A_out = get_plus_out_flux( v, rho, c, insideCellDir, norm ) * A;  

                                if ( !_no_species ){

                                  sum_in += rho_u_A_in * (_C-_A*phi[c]); 
                                  sum_out += rho_u_A_out* (_C-_A*phi[c]);  

                                } else {

                                  sum_in += rho_u_A_in; 
                                  sum_out += rho_u_A_out; 

                                }
                              }

                            }
                            break; 
                          case Patch::zminus: 
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              if ( cell_type[ c - insideCellDir] == -1 ){
                                const double A     = Dx.x()*Dx.y();     

                                int norm = 2; 

                                const double rho_u_A_in  = get_minus_in_flux( w, rho, c, insideCellDir, norm ) * A;  
                                const double rho_u_A_out = get_minus_out_flux( w, rho, c, insideCellDir, norm ) * A;  

                                if ( !_no_species ){

                                  sum_in += rho_u_A_in * (_C-_A*phi[c]); 
                                  sum_out += rho_u_A_out * (_C-_A*phi[c]); 

                                } else {

                                  sum_in += rho_u_A_in; 
                                  sum_out += rho_u_A_out; 

                                }
                              }

                            }
                            break; 
                          case Patch::zplus:
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              if ( cell_type[ c - insideCellDir] == -1 ){
                                const double A     = Dx.x()*Dx.y();     

                                int norm = 2; 

                                const double rho_u_A_in  = get_plus_in_flux( w, rho, c, insideCellDir, norm ) * A;  
                                const double rho_u_A_out = get_plus_out_flux( w, rho, c, insideCellDir, norm ) * A;  

                                if ( !_no_species ){

                                  sum_in += rho_u_A_in * (_C-_A*phi[c]); 
                                  sum_out += rho_u_A_out * (_C-_A*phi[c]); 

                                } else {

                                  sum_in += rho_u_A_in; 
                                  sum_out += rho_u_A_out; 

                                }
                              }
                            }
                            break; 
                          default: 
                            std::ostringstream msg; 
                            msg << "Error: Face type not recognized: " << face << std::endl;
                            throw InvalidValue(msg.str(), __FILE__, __LINE__); 
                            break; 
                        } 
                      }
                    }
                  }
                } 
              }

              new_dw->put( sum_vartype( sum_in ), _IN_label ); 
              new_dw->put( sum_vartype( sum_out ), _OUT_label ); 
              new_dw->put( sum_vartype( sum_accum ), _ACCUM_label ); 
              new_dw->put( sum_vartype( sum_source ), _SOURCE_label ); 

            }
          
          }; 

          /** @brief Should actually compute the efficiency */            
          void sched_computeEfficiency( const LevelP& level, 
                                        SchedulerP& sched )
          {
          
            const std::string name =  "MassBalance::computeEfficiency";
            Task* tsk = scinew Task( name, this, 
                &MassBalance::computeEfficiency); 

            tsk->requires( Task::NewDW, _IN_label ); 
            tsk->requires( Task::NewDW, _OUT_label ); 
            tsk->requires( Task::NewDW, _ACCUM_label ); 
            tsk->requires( Task::NewDW, _SOURCE_label ); 

            tsk->computes( _residual_label ); 

            sched->addTask( tsk, level->eachPatch(), _a_labs->d_sharedState->allArchesMaterials() ); 
          
          };

          void computeEfficiency(  const ProcessorGroup* pc, 
                                   const PatchSubset* patches, 
                                   const MaterialSubset* matls, 
                                   DataWarehouse* old_dw, 
                                   DataWarehouse* new_dw )
          {

            sum_vartype in; 
            sum_vartype out; 
            sum_vartype accum; 
            sum_vartype source; 
            new_dw->get( in, _IN_label ); 
            new_dw->get( out, _OUT_label ); 
            new_dw->get( accum, _ACCUM_label );
            new_dw->get( source, _SOURCE_label ); 

            double residual = out - in + accum - source; 
  
            new_dw->put( delt_vartype( residual ), _residual_label ); 

          }; 

        private: 

          template<typename UT>
          double inline get_minus_in_flux( UT                      & u,
                                           constCCVariable<double> & rho,
                                           const IntVector         & c, 
                                           const IntVector         & inside_dir,
                                           const int                 norm ){ 

            IntVector cp = c - inside_dir; 

            const double rho_f = 0.5 * ( rho[c] + rho[cp] ); 
            const double u_f   = std::abs(std::min( 0.0, inside_dir[norm]*u[c] ));

            const double flux = rho_f * u_f; 

            return flux; 

          }

          template<typename UT>
          double inline get_minus_out_flux( UT                      & u,
                                            constCCVariable<double> & rho,
                                            const IntVector         & c, 
                                            const IntVector         & inside_dir,
                                            const int                 norm ){ 

            IntVector cp = c - inside_dir; 

            const double rho_f = 0.5 * ( rho[c] + rho[cp] ); 
            const double u_f   = std::max( 0.0, inside_dir[norm]*u[c] ); 

            const double flux = rho_f * u_f; 

            return flux; 

          }

          template<typename UT>
          double inline get_plus_in_flux( UT                      & u,
                                          constCCVariable<double> & rho,
                                          const IntVector         & c, 
                                          const IntVector         & inside_dir,
                                          const int                  norm ){  

            IntVector cp = c - inside_dir; 

            const double rho_f = 0.5 * ( rho[c] + rho[cp] ); 
            const double u_f   = std::abs(std::min( 0.0, inside_dir[norm]*u[cp] )); 

            const double flux = rho_f * u_f; 

            return flux; 

          }

          template<typename UT>
          double inline get_plus_out_flux( UT                      & u,
                                           constCCVariable<double> & rho,
                                           const IntVector         & c, 
                                           const IntVector         & inside_dir, 
                                           const int                 norm ){ 
            
            IntVector cp = c - inside_dir; 

            const double rho_f = 0.5 * ( rho[c] + rho[cp] ); 
            const double u_f   = std::max( 0.0, inside_dir[norm]*u[cp] ); 

            const double flux = rho_f * u_f; 

            return flux; 

          }

          const VarLabel* _IN_label; 
          const VarLabel* _OUT_label; 
          const VarLabel* _ACCUM_label; 
          const VarLabel* _SOURCE_label; 
          const VarLabel* _residual_label; 
          const VarLabel* _phi_label; 
          const VarLabel* _S_label; 

          const BoundaryCondition* _bcs; 

          bool   _no_species; 
          bool   _no_source; 

          double _A;
          double _C; 

      }; 

      //_______________COMBUSTION EFFICIENCY_____________________
      class CombustionEfficiency : public Calculator { 

        /** @class  Combustion Efficiency calculator
         *  @author Jeremy Thornock
         *  @date   March, 2012
         *
         *  @brief Computes the combustion efficiency. 
         *
         *  @details 
         * Computes
         *
         * \eta = 1 - \frac{\int \phi f w H(w) dA}{\int f w H(w) dA}
         *
         * where 
         *
         * \phi = some balance variable
         * w    = mass flus (rho*u)
         * f    = mixture fraction 
         * A    = cell face area 
         * H(w) = Heavyside function 
         *
         * Note that in some cases, f may be a sum of two mixture fractions. 
         *
         * The following spec holds: 
         *
         *
         *
         *
         */

        public: 

          CombustionEfficiency(std::string id, const BoundaryCondition* bcs, ArchesLabel* a_lab) 
            : Calculator(id, a_lab), _bcs(bcs) {

            proc0cout << " Instantiating a calculator named: " << _id << " of type: combustion_efficiency" << std::endl;

            _numerator_label = VarLabel::create(   _id+"_numerator", sum_vartype::getTypeDescription() ); 
            _denominator_label = VarLabel::create( _id+"_denominator", sum_vartype::getTypeDescription() ); 
            _efficiency_label = VarLabel::create(  _id, min_vartype::getTypeDescription() ); 
          
          }
          ~CombustionEfficiency(){
          
            VarLabel::destroy( _numerator_label ); 
            VarLabel::destroy( _denominator_label ); 
            VarLabel::destroy( _efficiency_label ); 

          }

          bool problemSetup( const ProblemSpecP& db ){

            ProblemSpecP params = db; 
         
            double N; 
            db->findBlock("mixture_fraction")->getAttribute("N",N); 
            _num_mf = int(N); 

            if ( _num_mf == 1 ) { 

              db->findBlock("mixture_fraction")->getAttribute("mf_label_1", _mf_id_1); 

              _mf_1_label = VarLabel::find( _mf_id_1 ); 

            } else if ( _num_mf == 2 ) { 

              db->findBlock("mixture_fraction")->getAttribute("mf_label_1", _mf_id_1); 
              db->findBlock("mixture_fraction")->getAttribute("mf_label_2", _mf_id_2); 

              _mf_1_label = VarLabel::find( _mf_id_1 ); 
              _mf_2_label = VarLabel::find( _mf_id_2 ); 

            } else { 

              throw InvalidValue("Error: Number of mixture fractions for combustion_efficiency can only be 1 or 2.",__FILE__, __LINE__); 

            } 

            std::string phi_name; 
            params->require("phi_label", phi_name); 
            _phi_label = VarLabel::find( phi_name ); 

            _phi_at_feq1 = 0.0; 
            params->require("phi_at_feq1", _phi_at_feq1); 

            return true; 
          
          }

          /** @brief Should compute any summation over boundaries */ 
          void sched_computeReductionVars( const LevelP& level, 
                                           SchedulerP& sched ){
        
            const std::string name =  "CombustionEfficicency::computeReductionVars";
            Task* tsk = scinew Task( name, this, 
                &CombustionEfficiency::computeReductionVars); 

            tsk->computes( _numerator_label ); 
            tsk->computes( _denominator_label ); 

            if ( _num_mf == 1 ){ 

              tsk->requires( Task::NewDW, _mf_1_label, Ghost::None, 0); 

            } else if ( _num_mf == 2 ){ 
          
              tsk->requires( Task::NewDW, _mf_1_label, Ghost::None, 0); 
              tsk->requires( Task::NewDW, _mf_2_label, Ghost::None, 0); 

            }

            tsk->requires( Task::NewDW, _a_labs->d_densityCPLabel, Ghost::None, 0 ); 
            tsk->requires( Task::NewDW, _a_labs->d_uVelocitySPBCLabel, Ghost::None, 0 ); 
            tsk->requires( Task::NewDW, _a_labs->d_vVelocitySPBCLabel, Ghost::None, 0 ); 
            tsk->requires( Task::NewDW, _a_labs->d_wVelocitySPBCLabel, Ghost::None, 0 ); 
            tsk->requires( Task::NewDW, _phi_label, Ghost::None, 0 ); 

            sched->addTask( tsk, level->eachPatch(), _a_labs->d_sharedState->allArchesMaterials() ); 
          
          }

          void computeReductionVars( const ProcessorGroup* pc, 
                                     const PatchSubset* patches, 
                                     const MaterialSubset* matls, 
                                     DataWarehouse* old_dw, 
                                     DataWarehouse* new_dw )
          {
            for (int p = 0; p < patches->size(); p++) {

              const Patch* patch = patches->get(p);
              const Level* level = patch->getLevel(); 
              const int ilvl = level->getID(); 
              int archIndex = 0; // only one arches material
              int indx = _a_labs->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

              constCCVariable<double> mf_1; 
              constCCVariable<double> mf_2; 
              constSFCXVariable<double> u; 
              constSFCYVariable<double> v; 
              constSFCZVariable<double> w; 
              constCCVariable<double> rho; 
              constCCVariable<double> phi; 

              if ( _num_mf == 1 ){ 

                new_dw->get( mf_1, _mf_1_label, indx, patch, Ghost::None, 0 ); 

              } else if ( _num_mf == 2 ){ 

                new_dw->get( mf_1, _mf_1_label, indx, patch, Ghost::None, 0 ); 
                new_dw->get( mf_2, _mf_2_label, indx, patch, Ghost::None, 0 ); 

              } 

              new_dw->get( u, _a_labs->d_uVelocitySPBCLabel, indx, patch, Ghost::None, 0 ); 
              new_dw->get( v, _a_labs->d_vVelocitySPBCLabel, indx, patch, Ghost::None, 0 ); 
              new_dw->get( w, _a_labs->d_wVelocitySPBCLabel, indx, patch, Ghost::None, 0 ); 
              new_dw->get( rho, _a_labs->d_densityCPLabel, indx, patch, Ghost::None, 0 ); 
              new_dw->get( phi, _phi_label, indx, patch, Ghost::None, 0 ); 

              double sum_num = 0.0;
              double sum_den = 0.0; 

              for ( BoundaryCondition::BCInfoMap::const_iterator bc_iter = _bcs->d_bc_information[ilvl].begin(); 
                    bc_iter != _bcs->d_bc_information[ilvl].end(); bc_iter++){

                if ( bc_iter->second.type == BoundaryCondition::OUTLET || 
                     bc_iter->second.type == BoundaryCondition::PRESSURE ){ 

                  std::vector<Patch::FaceType>::const_iterator bf_iter;
                  std::vector<Patch::FaceType> bf;
                  patch->getBoundaryFaces(bf);

                  for (bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++){

                    //get the face
                    Patch::FaceType face = *bf_iter;
                    IntVector insideCellDir = patch->faceDirection(face); 
                    Vector Dx = patch->dCell(); 

                    //get the number of children
                    int numChildren = patch->getBCDataArray(face)->getNumberChildren(indx); //assumed one material

                    for (int child = 0; child < numChildren; child++){

                      double bc_value = 0;
                      //int norm = getNormal( face ); 
                      
                      std::string bc_kind = "NotSet";
                      Iterator bound_ptr;

                      bool foundIterator = 
                        getIteratorBCValueBCKind( patch, face, child, bc_iter->second.name, indx, bc_value, bound_ptr, bc_kind); 

                      if ( foundIterator ) {

                        switch(face) { 

                          case Patch::xminus: 
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              const double A     = Dx.y()*Dx.z();     
                              const double rho_u_A = get_minus_flux( u, rho, c, insideCellDir ) * A;  

                              if ( _num_mf == 1 ){ 
                                
                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * mf_1[c] * rho_u_A; 

                              } else { 

                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * ( mf_1[c] + mf_2[c] ) * rho_u_A; 

                              } 
                            }
                            break; 
                          case Patch::xplus:
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              const double A     = Dx.y()*Dx.z();     
                              const double rho_u_A = get_plus_flux( u, rho, c, insideCellDir ) * A;  
                              if ( _num_mf == 1 ){ 
                                
                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * mf_1[c] * rho_u_A; 

                              } else { 

                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * ( mf_1[c] + mf_2[c] ) * rho_u_A;

                              } 

                            }
                            break; 
                          case Patch::yminus: 
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              const double A     = Dx.x()*Dx.z();     
                              const double rho_u_A = get_minus_flux( v, rho, c, insideCellDir ) * A;  
                              if ( _num_mf == 1 ){ 
                                
                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * mf_1[c] * rho_u_A; 

                              } else { 

                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * ( mf_1[c] + mf_2[c] ) * rho_u_A; 

                              } 

                            }
                            break; 
                          case Patch::yplus:
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              const double A     = Dx.x()*Dx.z();     
                              const double rho_u_A = get_plus_flux( v, rho, c, insideCellDir ) * A;  
                              if ( _num_mf == 1 ){ 
                                
                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * mf_1[c] * rho_u_A; 

                              } else { 

                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * ( mf_1[c] + mf_2[c] ) * rho_u_A; 

                              } 

                            }
                            break; 
                          case Patch::zminus: 
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              const double A     = Dx.x()*Dx.y();     
                              const double rho_u_A = get_minus_flux( w, rho, c, insideCellDir ) * A;  
                              if ( _num_mf == 1 ){ 
                                
                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * mf_1[c] * rho_u_A; 

                              } else { 

                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * ( mf_1[c] + mf_2[c] ) * rho_u_A; 

                              } 

                            }
                            break; 
                          case Patch::zplus:
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              const double A     = Dx.x()*Dx.y();     
                              const double rho_u_A = get_plus_flux( w, rho, c, insideCellDir ) * A;  
                              if ( _num_mf == 1 ){ 
                                
                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * mf_1[c] * rho_u_A; 

                              } else { 

                                sum_num += phi[c] * rho_u_A; 
                                sum_den += _phi_at_feq1 * ( mf_1[c] + mf_2[c] ) * rho_u_A; 

                              } 

                            }
                            break; 
                          default: 
                            std::ostringstream msg; 
                            msg << "Error: Face type not recognized: " << face << std::endl;
                            throw InvalidValue(msg.str(), __FILE__, __LINE__); 
                            break; 
                        } 
                      }
                    }
                  }
                } 
              }

              new_dw->put( sum_vartype( sum_num ), _numerator_label ); 
              new_dw->put( sum_vartype( sum_den ), _denominator_label ); 

            }
          
          }

          /** @brief Should actually compute the efficiency */            
          void sched_computeEfficiency( const LevelP& level, 
                                        SchedulerP& sched )
          {
          
            const std::string name =  "CombustionEfficicency::computeEfficiency";
            Task* tsk = scinew Task( name, this, 
                &CombustionEfficiency::computeEfficiency); 

            tsk->requires( Task::NewDW, _numerator_label ); 
            tsk->requires( Task::NewDW, _denominator_label ); 

            tsk->computes( _efficiency_label ); 

            sched->addTask( tsk, level->eachPatch(), _a_labs->d_sharedState->allArchesMaterials() ); 
          
          }

          void computeEfficiency(  const ProcessorGroup* pc, 
                                   const PatchSubset* patches, 
                                   const MaterialSubset* matls, 
                                   DataWarehouse* old_dw, 
                                   DataWarehouse* new_dw )
          {

            sum_vartype numerator; 
            sum_vartype denominator; 
            new_dw->get( numerator, _numerator_label ); 
            new_dw->get( denominator, _denominator_label ); 
            const double small = 1.0e-16; 

            double combustion_efficiency = 0.0;
            if ( denominator > small ) { 
              if ( numerator > small ) { 
                combustion_efficiency = 1.0 - numerator / denominator; 
              } else { 
                combustion_efficiency = 1.0;
              } 
            } else { 
              combustion_efficiency = 1.0; 
            } 
  
            new_dw->put( delt_vartype( combustion_efficiency ), _efficiency_label ); 

          }

        private: 

          template<typename UT>
          double inline get_minus_flux( UT                      & u,
                                        constCCVariable<double> & rho,
                                        const IntVector         & c, 
                                        const IntVector         & inside_dir ){ 

            IntVector cp = c - inside_dir; 
            IntVector cm = c + inside_dir; 

            const double rho_f = 0.5 * ( rho[c] + rho[cp] ); 
            const double u_f   = std::min( 0.0, u[c] ); 

            const double flux = rho_f * u_f; 

            return flux; 
          }

          template<typename UT>
          double inline get_plus_flux( UT                      & u,
                                       constCCVariable<double> & rho,
                                       const IntVector         & c, 
                                       const IntVector         & inside_dir ){

            IntVector cp = c - inside_dir; 
            IntVector cm = c + inside_dir; 

            const double rho_f = 0.5 * ( rho[c] + rho[cp] ); 
            const double u_f   = std::max( 0.0, u[cp] ); 

            const double flux = rho_f * u_f; 

            return flux; 
          }

          std::string _mf_id_1; 
          std::string _mf_id_2; 

          int         _num_mf; 
          double      _phi_at_feq1; 

          const VarLabel* _numerator_label; 
          const VarLabel* _denominator_label; 
          const VarLabel* _efficiency_label; 
          const VarLabel* _mf_1_label; 
          const VarLabel* _mf_2_label; 
          const VarLabel* _phi_label;

          const BoundaryCondition* _bcs; 

      };


      //__________________________________________________

      typedef std::map<std::string, Calculator*> LOC; // List Of Calculators
      LOC _my_calculators; 

      const BoundaryCondition * _bcs; 
      ArchesLabel             * _a_labs; 

  }; 
}

#endif
