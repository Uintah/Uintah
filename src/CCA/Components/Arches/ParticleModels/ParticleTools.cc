#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <sstream>
#include <vector>


namespace Uintah{ namespace ArchesCore{

//--------------------------------------------------------------------------------------------------
  std::string parse_for_particle_role_to_label( ProblemSpecP& db, ArchesCore::PARTICLE_ROLE role ){

    const ProblemSpecP params_root = db->getRootNode();
    if ( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("EulerianParticles") ){

      if ( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("EulerianParticles")->findBlock("ParticleVariables") ){

        const ProblemSpecP db_pvar = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("EulerianParticles")->findBlock("ParticleVariables");

        for( ProblemSpecP db_var = db_pvar->findBlock("variable"); db_var != nullptr; db_var = db_var->findNextBlock("variable")){

          std::string label;
          std::string role_found;
          db_var->getAttribute("label", label);
          db_var->getAttribute("role", role_found);

          if ( ArchesCore::get_particle_role_from_enum(role) == role_found ){
            return label;
          }

        }

        throw ProblemSetupException("Error: This Eulerian particle role not found: "+
          ArchesCore::get_particle_role_from_enum(role), __FILE__, __LINE__);

      } else {

        throw ProblemSetupException("Error: No <EulerianParticles><ParticleVariables> section found.",__FILE__,__LINE__);

      }
    } else {

      throw ProblemSetupException("Error: No <EulerianParticles> section found in input file.",__FILE__,__LINE__);

    }

  }

//--------------------------------------------------------------------------------------------------
  std::string get_particle_role_from_enum( PARTICLE_ROLE role ){

    if ( role == P_SIZE ){
      return "size";
    } else if ( role == P_XVEL ){
      return "uvel";
    } else if ( role == P_YVEL ){
      return "vvel";
    } else if ( role == P_ZVEL ){
      return "wvel";
    } else if ( role == P_CHAR ){
      return "char";
    } else if ( role == P_RAWCOAL ){
      return "raw_coal";
    } else if ( role == P_DENSITY ){
      return "density";
    } else if ( role == P_ENTHALPY ){
      return "enthalpy";
    } else if ( role == P_TEMPERATURE ){
      return "temperature";
    } else if ( role == P_MAXTEMPERATURE ){
      return "max_temperature";
    } else if ( role == P_DTDT ){
      return "dTdt";
    } else if ( role == P_TOTNUM_DENSITY ){
      return "total_number_density";
    } else {
      throw InvalidValue("Error: No particle role found.", __FILE__, __LINE__ );
    }

  }

//--------------------------------------------------------------------------------------------------
  std::string append_env( std::string in_name, const int qn ){

    using namespace std;

    std::string new_name = in_name;
    std::stringstream str_qn;
    str_qn << qn;
    new_name += "_";
    new_name += str_qn.str();
    return new_name;

  }

//--------------------------------------------------------------------------------------------------
  std::string append_qn_env( std::string in_name, const int qn ){

    std::string new_name = in_name;
    std::stringstream str_qn;
    str_qn << qn;
    new_name += "_qn";
    new_name += str_qn.str();
    return new_name;

  }

//--------------------------------------------------------------------------------------------------
  bool check_for_particle_method( ProblemSpecP& db, PARTICLE_METHOD method ){

    const ProblemSpecP arches_root = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES");

    bool check = false;
    if ( method == DQMOM_METHOD ){

      if ( arches_root->findBlock("DQMOM") ){
        check = true;
      }

    } else if ( method == CQMOM_METHOD ){

      if ( arches_root->findBlock("CQMOM") ){
        check = true;
      }

    } else if ( method == LAGRANGIAN_METHOD ){

      if ( arches_root->findBlock("LagrangianParticles") ){
        check = true;
      }

    } else {
      throw ProblemSetupException("Error: Particle method type not recognized.", __FILE__, __LINE__);
    }

    return check;

  }

//--------------------------------------------------------------------------------------------------
  int get_num_env( ProblemSpecP& db, PARTICLE_METHOD method ){

    const ProblemSpecP arches_root = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES");

    if ( method == DQMOM_METHOD ){

      if ( arches_root->findBlock("DQMOM") ){
        int N;
        arches_root->findBlock("DQMOM")->require("number_quad_nodes",N);
        return N;
      } else {
        throw ProblemSetupException("Error: DQMOM particle method not found.", __FILE__, __LINE__);
      }

    } else if ( method == CQMOM_METHOD ){

      if ( arches_root->findBlock("CQMOM") ){
        int N = 1;
        std::vector<int> N_i;
        arches_root->findBlock("CQMOM")->require("QuadratureNodes",N_i);
        for (unsigned int i = 0; i < N_i.size(); i++ ) {
          N *= N_i[i];
        }
        return N;
      } else {
        throw ProblemSetupException("Error: CQMOM particle method not found.", __FILE__, __LINE__);
      }

    } else if ( method == LAGRANGIAN_METHOD ){

      if ( arches_root->findBlock("LagrangianParticles") ){
        return 1;
      } else {
        throw ProblemSetupException("Error: DQMOM particle method not found.", __FILE__, __LINE__);
      }

    } else {
      throw ProblemSetupException("Error: Particle method type not recognized.", __FILE__, __LINE__);
    }
  }

//--------------------------------------------------------------------------------------------------
  std::string get_model_type( ProblemSpecP& db, std::string model_name, PARTICLE_METHOD method ){

    const ProblemSpecP arches_root = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES");

    if ( method == DQMOM_METHOD ){

      const ProblemSpecP db_models = arches_root->findBlock("DQMOM")->findBlock("Models");

      for ( ProblemSpecP m_db = db_models->findBlock("model"); m_db != nullptr; m_db = m_db->findNextBlock("model") ) {

        std::string curr_model_name;
        std::string curr_model_type;
        m_db->getAttribute("label",curr_model_name);
        m_db->getAttribute("type", curr_model_type);

        if ( model_name == curr_model_name ){
          return curr_model_type;
        }
      }
    }
    else {
      throw InvalidValue("Error: Not yet implemented for this particle method.",__FILE__,__LINE__);
    }
    return "nullptr";
  }

//--------------------------------------------------------------------------------------------------
  double get_scaling_constant(ProblemSpecP& db, const std::string labelName, const int qn){

    const ProblemSpecP params_root = db->getRootNode();
    const ProblemSpecP db_dqmom = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM");
    if ( db_dqmom ){

      if ( labelName == "w" ||  labelName == "weights" || labelName == "weight"){

        if ( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->findBlock("Weights") ){

          std::vector<double> scaling_const;
          ProblemSpecP db_weight_scal = db_dqmom->findBlock("Weights")->findBlock("scaling_const");
          if ( db_weight_scal ) { 
            db_dqmom->findBlock("Weights")->require("scaling_const",scaling_const);
            return scaling_const[qn];
          }else {
            return 1.;
          }
        }
        else {
          throw ProblemSetupException("Error: cannot find <weights> block in inupt file.",__FILE__,__LINE__);
        }
      }
      else {
        for(  ProblemSpecP IcBlock =params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->findBlock("Ic"); IcBlock != nullptr;
        IcBlock = IcBlock->findNextBlock("Ic") ) {

          std::string tempLabelname;
          IcBlock->getAttribute("label",tempLabelname);
          if (tempLabelname == labelName){
            std::vector<double> scaling_const;
            ProblemSpecP db_ic_scal = IcBlock->findBlock("scaling_const");
            if ( db_ic_scal ) { 
              IcBlock->require("scaling_const",scaling_const);
              return scaling_const[qn];
            } else {
              return 1.;
            }
          }
        }
        //throw ProblemSetupException("Error: couldn't find internal coordinate or weight with name: "+labelName , __FILE__, __LINE__);
        return 1.;
      }
    } else {
      throw ProblemSetupException("Error: DQMOM section not found in input file.",__FILE__,__LINE__);
    }
  }

//--------------------------------------------------------------------------------------------------
  bool get_model_value(ProblemSpecP& db, const std::string labelName, const int qn, double &value){

    const ProblemSpecP params_root = db->getRootNode();
    if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleModels") ){
      const ProblemSpecP params_particleModels = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleModels");
      for ( ProblemSpecP modelBlock =params_particleModels->findBlock("model"); modelBlock != nullptr;
      modelBlock = modelBlock->findNextBlock("model") ) {

        std::string tempLabelName;
        modelBlock->getAttribute("label",tempLabelName);
        if (tempLabelName == labelName){
          // -- expand this as different internal coordinate models are added--//
          std::string tempTypeName;
          modelBlock->getAttribute("type",tempTypeName);

          if( tempTypeName== "constant"){
            std::vector<double> internalCoordinateValue;
            modelBlock->require("constant", internalCoordinateValue);
            value= internalCoordinateValue[qn];
            return true;
          }
        }
      }
    }
    return false;
  }

//--------------------------------------------------------------------------------------------------
  double get_inlet_particle_density(ProblemSpecP& db){

    const ProblemSpecP params_root = db->getRootNode();
    if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")){
      double density;
      params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->require("density",density);
      return density;
    }
    else {
      throw ProblemSetupException("Error: cannot find <ParticleProperties> in arches block.",__FILE__,__LINE__);
    }

    return false;
  }

//--------------------------------------------------------------------------------------------------
  double get_inlet_particle_size(ProblemSpecP& db, const int qn ){

    const ProblemSpecP params_root = db->getRootNode();
    if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("diameter_distribution")){
      std::vector<double> _sizes;
      params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->require("diameter_distribution", _sizes);
      double initial_size = _sizes[qn];
      return initial_size;
    } else{
      throw ProblemSetupException("Error: cannot find <diameter_distribution> in arches block.",__FILE__,__LINE__);
    }

    return false;
  }

//--------------------------------------------------------------------------------------------------
  std::vector<std::string> getICNames( ProblemSpecP& db ){

    const ProblemSpecP db_dqmom = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM");
    std::vector<std::string> ic_names;

    if ( db_dqmom ){

      for ( ProblemSpecP db_ic = db_dqmom->findBlock("Ic"); db_ic != nullptr;
      db_ic = db_ic->findNextBlock("Ic") ){

        std::string ic_name;
        db_ic->getAttribute("label", ic_name );
        ic_names.push_back(ic_name);

      }

    } else {

      std::stringstream msg;
      msg << "Error: Trying to look for internal coordinates but no DQMOM exists in input file."
      << std::endl;
      throw ProblemSetupException( msg.str(), __FILE__, __LINE__ );

    }

    return ic_names;

  }

//--------------------------------------------------------------------------------------------------
  std::vector<std::string> getICModels( ProblemSpecP& db, const std::string ic_name ){

    const ProblemSpecP db_dqmom
      = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM");
    std::vector<std::string> ic_models;

    if ( db_dqmom ){

      if ( ic_name == "w" ){

        ProblemSpecP db_weight = db_dqmom->findBlock("Weights");
        for ( ProblemSpecP db_model = db_weight->findBlock("model"); db_model != nullptr;
              db_model = db_model->findNextBlock("model") ){

          std::string this_model;
          db_model->getAttribute("label", this_model);
          ic_models.push_back(this_model);

        }

      } else {

        for ( ProblemSpecP db_ic = db_dqmom->findBlock("Ic"); db_ic != nullptr;
        db_ic = db_ic->findNextBlock("Ic") ){

          std::string label;
          db_ic->getAttribute("label", label );

          if ( ic_name == label ){

            for ( ProblemSpecP db_model = db_ic->findBlock("model"); db_model != nullptr;
            db_model = db_model->findNextBlock("model") ){
              std::string this_model;
              db_model->getAttribute("label", this_model);
              ic_models.push_back(this_model);
            }
          }
        }

      }

    } else {

      std::stringstream msg;
      msg << "Error: Trying to look for internal coordinates but no DQMOM exists in input file."
      << std::endl;
      throw ProblemSetupException( msg.str(), __FILE__, __LINE__ );

    }

    return ic_models;

  }

//--------------------------------------------------------------------------------------------------
 std::map<std::string, std::string> getModelTypeMap( ProblemSpecP& db, const std::string ic_name ){

     const ProblemSpecP db_dqmom
       = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM");
     std::vector<std::string> ic_models;

     std::map<std::string, std::string> the_map; // (type) -> (name of the model) FOR THIS IC
     the_map.clear();

     if ( db_dqmom ){

       ProblemSpecP db_models = db_dqmom->findBlock("Models");

       if ( ic_name == "w" ){

         ProblemSpecP db_weight = db_dqmom->findBlock("Weights");
         for ( ProblemSpecP db_model = db_weight->findBlock("model"); db_model != nullptr;
               db_model = db_model->findNextBlock("model") ){

           std::string this_model;
           db_model->getAttribute("label", this_model);

           //Now find the model in the list of <Models> to get the type:
           for ( ProblemSpecP db_m = db_models->findBlock("model"); db_m != nullptr;
                  db_m = db_m->findNextBlock("model")){

             std::string name;
             std::string type;
             db_m->getAttribute("label", name);
             db_m->getAttribute("type", type);
             if ( this_model == name ){
               the_map.insert(std::make_pair(type, name));
             }

           }
         }

       } else {

         for ( ProblemSpecP db_ic = db_dqmom->findBlock("Ic"); db_ic != nullptr;
         db_ic = db_ic->findNextBlock("Ic") ){

           std::string label;
           db_ic->getAttribute("label", label );

           if ( ic_name == label ){

             for ( ProblemSpecP db_model = db_ic->findBlock("model"); db_model != nullptr;
                   db_model = db_model->findNextBlock("model") ){

               std::string this_model;
               db_model->getAttribute("label", this_model);

               //Now find the model in the list of <Models> to get the type:
               for ( ProblemSpecP db_m = db_models->findBlock("model"); db_m != nullptr;
                      db_m = db_m->findNextBlock("model")){

                 std::string name;
                 std::string type;
                 db_m->getAttribute("label", name);
                 db_m->getAttribute("type", type);
                 if ( this_model == name ){
                   the_map.insert(std::make_pair(type, name));
                 }
               }
             }
           }
         }
       }

     } else {

       std::stringstream msg;
       msg << "Error: Trying to look for internal coordinates but no DQMOM exists in input file."
       << std::endl;
       throw ProblemSetupException( msg.str(), __FILE__, __LINE__ );

     }

     return the_map;

 }

 std::string getModelNameByType( ProblemSpecP& db, const std::string ic_name,
                                 const std::string model_type ){

   std::map<std::string, std::string> the_map = getModelTypeMap( db, ic_name );
   auto i = the_map.find( model_type );
   if ( i == the_map.end() ){
     return "NULLSTRING";
   } else {
     return i->second;
   }
 }

 }} //Uintah::ArchesCore
