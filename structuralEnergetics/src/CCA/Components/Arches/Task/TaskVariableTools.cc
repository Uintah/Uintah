#include <CCA/Components/Arches/Task/FieldContainer.h>
#include <CCA/Components/Arches/Task/TaskVariableTools.h>

namespace Uintah {

   void register_variable_work( std::string name,
                                ArchesFieldContainer::VAR_DEPEND dep,
                                int nGhost,
                                ArchesFieldContainer::WHICH_DW dw,
                                ArchesFieldContainer::VariableRegistry& variable_registry,
                                const int time_substep,
                                const std::string task_name )
   {

    ArchesFieldContainer::VariableInformation info;

    info.name   = name;
    info.depend = dep;
    info.dw     = dw;
    info.nGhost = nGhost;
    info.local = false;

    info.is_constant = false;
    if ( dep == ArchesFieldContainer::REQUIRES ){
      info.is_constant = true;
    }

    switch (dw) {

    case ArchesFieldContainer::OLDDW:

      info.uintah_task_dw = Task::OldDW;
      break;

    case ArchesFieldContainer::NEWDW:

      info.uintah_task_dw = Task::NewDW;
      break;

    case ArchesFieldContainer::LATEST:

      if ( time_substep == 0 ){
        info.dw = ArchesFieldContainer::OLDDW;
        info.uintah_task_dw = Task::OldDW;
      } else {
        info.dw = ArchesFieldContainer::NEWDW;
        info.uintah_task_dw = Task::NewDW;
      }
      break;

    default:

      throw InvalidValue("Arches Task Error: Cannot determine the DW needed for variable: "+name, __FILE__, __LINE__);
      break;

    }

    //check for conflicts:
    if (dep == ArchesFieldContainer::COMPUTES && dw == ArchesFieldContainer::OLDDW) {
      throw InvalidValue("Arches Task Error: Cannot COMPUTE (ArchesFieldContainer::COMPUTES) a variable from OldDW for variable: "+name, __FILE__, __LINE__);
    }

    if ( (dep == ArchesFieldContainer::MODIFIES && dw == ArchesFieldContainer::OLDDW) ) {
      throw InvalidValue("Arches Task Error: Cannot MODIFY a variable from OldDW for variable: "+name, __FILE__, __LINE__);
    }

    if ( dep == ArchesFieldContainer::COMPUTES || dep == ArchesFieldContainer::MODIFIES ){

      if ( nGhost > 0 ) {

        std::cout << "Arches Task Warning: A variable COMPUTE/MODIFIES found that is requesting ghosts for: "+name+" Nghosts set to zero!" << std::endl;
        info.nGhost = 0;

      }
    }

    const VarLabel* the_label = nullptr;
    the_label = VarLabel::find( name );

    if ( the_label == nullptr ){
      throw InvalidValue("Error: The variable named: "+name+" does not exist for task: "+task_name,__FILE__,__LINE__);
    } else {
      info.label = the_label;
    }

    const Uintah::TypeDescription* type_desc = the_label->typeDescription();

    if ( dep == ArchesFieldContainer::REQUIRES ) {

      if ( nGhost == 0 ){
        info.ghost_type = Ghost::None;
      } else {
        if ( type_desc == CCVariable<int>::getTypeDescription() ) {
            info.ghost_type = Ghost::AroundCells;
        } else if ( type_desc == CCVariable<double>::getTypeDescription() ) {
            info.ghost_type = Ghost::AroundCells;
        } else if ( type_desc == CCVariable<Vector>::getTypeDescription() ) {
            info.ghost_type = Ghost::AroundCells;
        } else if ( type_desc == SFCXVariable<double>::getTypeDescription() ) {
            info.ghost_type = Ghost::AroundFaces;
        } else if ( type_desc == SFCYVariable<double>::getTypeDescription() ) {
            info.ghost_type = Ghost::AroundFaces;
        } else if ( type_desc == SFCZVariable<double>::getTypeDescription() ) {
            info.ghost_type = Ghost::AroundFaces;
        } else {
          throw InvalidValue("Error: No coverage yet for this type of variable.", __FILE__,__LINE__);
        }
      }
    }

    variable_registry.push_back( info );

  }

  void register_variable( std::string name,
                          ArchesFieldContainer::VAR_DEPEND dep,
                          int nGhost,
                          ArchesFieldContainer::WHICH_DW dw,
                          std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                          const int time_substep,
                          std::string task_name ){
    register_variable_work( name, dep, nGhost, dw, var_reg, time_substep, task_name );
  }

  void register_variable( std::string name,
                          ArchesFieldContainer::VAR_DEPEND dep,
                          int nGhost,
                          ArchesFieldContainer::WHICH_DW dw,
                          std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                          std::string task_name){
    register_variable_work( name, dep, nGhost, dw, var_reg, 0, task_name );
  }

  void register_variable( std::string name,
                          ArchesFieldContainer::VAR_DEPEND dep,
                          std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                          std::string task_name ){

    ArchesFieldContainer::WHICH_DW dw = ArchesFieldContainer::NEWDW;
    int nGhost = 0;
    register_variable_work( name, dep, nGhost, dw, var_reg, 0, task_name );

  }

  void register_variable( std::string name,
                          ArchesFieldContainer::VAR_DEPEND dep,
                          std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                          const int timesubstep,
                          std::string task_name){

    ArchesFieldContainer::WHICH_DW dw = ArchesFieldContainer::NEWDW;
    int nGhost = 0;
    register_variable_work( name, dep, nGhost, dw, var_reg, timesubstep, task_name );
  }
} // namespace Uintah
