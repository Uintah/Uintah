#ifndef Uintah_Component_Arches_TaskController_h
#define Uintah_Component_Arches_TaskController_h

namespace Uintah{ namespace ArchesCore {

class TaskController{

public:

  /** @brief Singleton Task Controller  **/
  static TaskController& self(){
    static TaskController s;
    return s;
  }

  void parse_task_controller( ProblemSpecP db_in ){

    //Parsed from <ARCHES>
    const ProblemSpecP params_root = db_in->getRootNode();
    ProblemSpecP db = params_root->findBlock("CFD")->findBlock("ARCHES");

    // This is a generic, global parameter for task packing.
    // One may want logic with more fine-grained control in
    // some cases. Default is false.
    ProblemSpecP db_controller = db->findBlock("TaskController");
    if ( db_controller != nullptr ){
      ProblemSpecP db_pack = db_controller->findBlock("TaskPacking");
      if ( db_pack != nullptr ){
        if ( db_pack->findBlock("global") ) packed_info.global = true;
        if ( db_pack->findBlock("turbulence") ) packed_info.turbulence = true;
        if ( db_pack->findBlock("scalar_transport") ) packed_info.scalar_transport= true;
        if ( db_pack->findBlock("momentum_transport") ) packed_info.momentum_transport= true;
      }
    }

  }


  /** @brief Contains switches to turn on/off packing of grouped tasks **/
  struct Packing{
    bool global{false};
    bool turbulence{false};
    bool scalar_transport{false};
    bool momentum_transport{false};
  };

  /** @brief Return the packing information **/
  const Packing& get_packing_info(){ return packed_info; }

private:

  TaskController(){}
  ~TaskController(){}

  Packing packed_info;

};  //class TaskController

}} //namepsace Uintah::ArchesCore





#endif
