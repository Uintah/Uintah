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
      void sched_computeEfficiencies( const LevelP& level, 
                                      SchedulerP& sched ) { 

        // loop through all calculators
        for ( LOC::iterator i = _my_calculators.begin(); i != _my_calculators.end(); i++ ){ 

          i->second->sched_computeReductionVars( level, sched ); 

          i->second->sched_computeEfficiency( level, sched ); 

        } 

      };

      /** @brief Scheduler for computing the various efficiencies */
      void sched_dummySolve( const LevelP& level, 
                             SchedulerP& sched ) { 

        // loop through all calculators
        for ( LOC::iterator i = _my_calculators.begin(); i != _my_calculators.end(); i++ ){ 

          i->second->sched_dummySolve( level, sched ); 

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

          void sched_dummySolve( const LevelP& level, 
                                  SchedulerP& sched )
          { 

            const std::string name =  "Calculator::dummySolve";
            Task* tsk = scinew Task( name, this, 
                &Calculator::dummySolve); 

            const VarLabel* label = VarLabel::find( _id ); 
            tsk->computes( label ); 

            sched->addTask( tsk, level->eachPatch(), _a_labs->d_sharedState->allArchesMaterials() ); 

          }; 

          void dummySolve( const ProcessorGroup* pc, 
                           const PatchSubset* patches, 
                           const MaterialSubset* matls, 
                           DataWarehouse* old_dw, 
                           DataWarehouse* new_dw )
          {

            const VarLabel* my_label = VarLabel::find(_id); 
            new_dw->put(delt_vartype(0.0), my_label); 

          };

        protected: 
          std::string _id; 
          ArchesLabel* _a_labs; 

          //.....

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

            std::cout << " Instantiating a calculator named: " << _id << " of type: combustion_efficiency" << std::endl;

            _numerator_label = VarLabel::create(   _id+"_numerator", sum_vartype::getTypeDescription() ); 
            _denominator_label = VarLabel::create( _id+"_denominator", sum_vartype::getTypeDescription() ); 
            _efficiency_label = VarLabel::create(  _id, min_vartype::getTypeDescription() ); 
          
          }; 
          ~CombustionEfficiency(){
          
            VarLabel::destroy( _numerator_label ); 
            VarLabel::destroy( _denominator_label ); 
            VarLabel::destroy( _efficiency_label ); 

          }; 

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

            return true; 
          
          }; 

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
              int archIndex = 0; // only one arches material
              int indx = _a_labs->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

              double numerator = 0.0; 
              double denominator = 0.0; 

              constCCVariable<double> mf_1; 
              constCCVariable<double> mf_2; 
              constSFCXVariable<double> u; 
              constSFCYVariable<double> v; 
              constSFCZVariable<double> w; 
              constCCVariable<double> rho; 

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

              double sum_num = 0.0;
              double sum_den = 0.0; 

              for ( BoundaryCondition::BCInfoMap::const_iterator bc_iter = _bcs->d_bc_information.begin(); 
                    bc_iter != _bcs->d_bc_information.end(); bc_iter++){

                if ( bc_iter->second.type == BoundaryCondition::OUTLET || 
                     bc_iter->second.type == BoundaryCondition::PRESSURE ){ 

                  vector<Patch::FaceType>::const_iterator bf_iter;
                  vector<Patch::FaceType> bf;
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
                      
                      string bc_kind = "NotSet";
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
                                
                                sum_num += mf_1[c] * rho_u_A; 
                                sum_den += rho_u_A; 

                              } else { 

                                sum_num += ( mf_1[c] + mf_2[c] ) * rho_u_A; 
                                sum_den += rho_u_A; 

                              } 
                            }
                            break; 
                          case Patch::xplus:
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              const double A     = Dx.y()*Dx.z();     
                              const double rho_u_A = get_plus_flux( u, rho, c, insideCellDir ) * A;  

                              if ( _num_mf == 1 ){ 
                                
                                sum_num += mf_1[c] * rho_u_A; 
                                sum_den += rho_u_A; 

                              } else { 

                                sum_num += ( mf_1[c] + mf_2[c] ) * rho_u_A; 
                                sum_den += rho_u_A; 

                              } 
                            }
                            break; 
                          case Patch::yminus: 
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              const double A     = Dx.x()*Dx.z();     
                              const double rho_u_A = get_minus_flux( v, rho, c, insideCellDir ) * A;  

                              if ( _num_mf == 1 ){ 
                                
                                sum_num += mf_1[c] * rho_u_A; 
                                sum_den += rho_u_A; 

                              } else { 

                                sum_num += ( mf_1[c] + mf_2[c] ) * rho_u_A; 
                                sum_den += rho_u_A; 

                              } 
                            }
                            break; 
                          case Patch::yplus:
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              const double A     = Dx.x()*Dx.z();     
                              const double rho_u_A = get_plus_flux( v, rho, c, insideCellDir ) * A;  

                              if ( _num_mf == 1 ){ 
                                
                                sum_num += mf_1[c] * rho_u_A; 
                                sum_den += rho_u_A; 

                              } else { 

                                sum_num += ( mf_1[c] + mf_2[c] ) * rho_u_A; 
                                sum_den += rho_u_A; 

                              } 
                            }
                            break; 
                          case Patch::zminus: 
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              const double A     = Dx.x()*Dx.y();     
                              const double rho_u_A = get_minus_flux( w, rho, c, insideCellDir ) * A;  

                              if ( _num_mf == 1 ){ 
                                
                                sum_num += mf_1[c] * rho_u_A; 
                                sum_den += rho_u_A; 

                              } else { 

                                sum_num += ( mf_1[c] + mf_2[c] ) * rho_u_A; 
                                sum_den += rho_u_A; 

                              } 
                            }
                            break; 
                          case Patch::zplus:
                            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                              IntVector c = *bound_ptr; 
                              const double A     = Dx.x()*Dx.y();     
                              const double rho_u_A = get_plus_flux( w, rho, c, insideCellDir ) * A;  

                              if ( _num_mf == 1 ){ 
                                
                                sum_num += mf_1[c] * rho_u_A; 
                                sum_den += rho_u_A; 

                              } else { 

                                sum_num += ( mf_1[c] + mf_2[c] ) * rho_u_A; 
                                sum_den += rho_u_A; 

                              } 
                            }
                            break; 
                          default: 
                            throw InvalidValue("Error: Face type not recognized: " + face, __FILE__, __LINE__); 
                            break; 
                        } 
                      }
                    }
                  }
                } 
              }

              new_dw->put( sum_vartype( numerator ), _numerator_label ); 
              new_dw->put( sum_vartype( denominator ), _denominator_label ); 

            }
          
          }; 

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
          
          };

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

            double combustion_efficiency = 0.0;
            if ( denominator > 0.0 ) { 
              combustion_efficiency = 1 - numerator / denominator; 
            }

            new_dw->put( delt_vartype( combustion_efficiency ), _efficiency_label ); 

          }; 

        private: 

          template<typename UT>
          const double inline get_minus_flux( UT& u, constCCVariable<double>& rho, const IntVector c, 
              const IntVector inside_dir ){ 

            IntVector cp = c - inside_dir; 
            IntVector cm = c + inside_dir; 

            const double rho_f = 0.5 * ( rho[c] + rho[cp] ); 
            const double u_f   = std::min( 0.0, u[c] ); 

            const double flux = rho_f * u_f; 

            return flux; 

          };

          template<typename UT>
          const double inline get_plus_flux( UT& u, constCCVariable<double>& rho, const IntVector c, 
              const IntVector inside_dir ){ 

            IntVector cp = c - inside_dir; 
            IntVector cm = c + inside_dir; 

            const double rho_f = 0.5 * ( rho[c] + rho[cp] ); 
            const double u_f   = std::max( 0.0, u[cp] ); 

            const double flux = rho_f * u_f; 

            return flux; 

          };

          std::string _mf_id_1; 
          std::string _mf_id_2; 

          int _num_mf; 

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

      const BoundaryCondition* _bcs; 
      ArchesLabel* _a_labs; 


  }; 

}

#endif
