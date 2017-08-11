#include <CCA/Components/Arches/Task/FieldContainer.h>
#include <CCA/Components/Arches/Task/TaskVariableTools.h>
#include <Core/Util/DebugStream.h>

static Uintah::DebugStream taskvar_dbgstream("ARCHES_TASK_VARS", false);
#define DBG_ON  taskvar_dbgstream.active()
#define DBG_STREAM  if( DBG_ON  ) taskvar_dbgstream

namespace Uintah {


   void register_variable_work( std::string name,
                                ArchesFieldContainer::VAR_DEPEND dep,
                                int nGhost,
                                ArchesFieldContainer::WHICH_DW dw,
                                ArchesFieldContainer::VariableRegistry& variable_registry,
                                const int time_substep,
                                const std::string task_name )
  {

    DBG_STREAM << " For task: " << task_name << " registering: " << std::endl;
    DBG_STREAM << "             name: " << name << std::endl;
    DBG_STREAM << "              dep: " << dep << std::endl;
    DBG_STREAM << "           nGhost: " << nGhost << std::endl;
    DBG_STREAM << "         which dw: " << dw << std::endl;
    DBG_STREAM << "     time_substep: " << time_substep << std::endl;

    if ( dw == ArchesFieldContainer::LATEST ){
      if ( time_substep == 0 ){
        dw = ArchesFieldContainer::OLDDW;
      } else {
        dw = ArchesFieldContainer::NEWDW;
      }
    }

    /// ERROR CHECKING ///
    bool add_variable = true;

    for ( auto i = variable_registry.begin(); i != variable_registry.end(); i++ ){

      //check if this name is already in the list:
      if ( (*i).name == name ){

        //does it have the same dependency?
        if ( (*i).depend == dep ){

          //Are they from the same DW?
          if ( (*i).dw == dw ) {

            if ( dep == ArchesFieldContainer::REQUIRES ){

              //default to the larger ghost requirement:
              if ( nGhost > (*i).nGhost ){

                //just modify the existing entry
                (*i).nGhost = nGhost;

              }

            }

            add_variable = false;

          }

          // else they are from different DataWarehouses
          // so go ahead and add this new variable

        }

        //is there a computes/modifies and requires from the same DW for the same variable?
        //this is disallowed.
        if ( dw == ArchesFieldContainer::NEWDW && (*i).dw == ArchesFieldContainer::NEWDW ){

          std::stringstream msg;
          msg << "\n** Warning: Potential task dependency problem! **\n " << "The variable: " << name
          << " is both required and modified in the same task:  " << task_name << std::endl <<
                 "This only represents a problem for the task if the required variable must remain constant. \n"
          <<     "Otherwise, it *may* be a feature of the task (e.g., this is by design). Best to check with the author of this task. \n \n";

          if ( dep == ArchesFieldContainer::REQUIRES && (*i).depend == ArchesFieldContainer::MODIFIES ){

            //throw InvalidValue(msg.str(), __FILE__, __LINE__ );
            proc0cout << msg.str();

          } else if ( dep == ArchesFieldContainer::MODIFIES && (*i).depend == ArchesFieldContainer::REQUIRES ){

            //throw InvalidValue(msg.str(), __FILE__, __LINE__ );
            proc0cout << msg.str();

          }

          std::stringstream msg_kill;
          msg_kill << "Error: The task, " << task_name << ", is attempting to compute a variable: " <<
                       name << " that is already required for this (or some other if tasks are packed) task." << std::endl;
          if ( dep == ArchesFieldContainer::COMPUTES && (*i).depend == ArchesFieldContainer::REQUIRES ){
            throw InvalidValue(msg_kill.str(), __FILE__, __LINE__ );
          } else if ( dep == ArchesFieldContainer::REQUIRES && (*i).depend == ArchesFieldContainer::COMPUTES ){
            if ( nGhost > 0 ){
              throw InvalidValue(msg_kill.str(), __FILE__, __LINE__ );
            } else {
              add_variable = false;
            }
          }

          //A variable is computed and modified
          // resort to a computes
          if ( dep == ArchesFieldContainer::COMPUTES && (*i).depend == ArchesFieldContainer::MODIFIES ){

            //just modify the information
            (*i).depend = ArchesFieldContainer::COMPUTES;
            add_variable = false;

          } else if ( dep == ArchesFieldContainer::MODIFIES  && (*i).depend == ArchesFieldContainer::COMPUTES ){

            //no need to add this variable
            add_variable = false;

          }

        }
      }
    }

    if ( add_variable ){

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
  }

  void register_variable( std::string name,
                          ArchesFieldContainer::VAR_DEPEND dep,
                          int nGhost,
                          ArchesFieldContainer::WHICH_DW dw,
                          std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                          const int time_substep,
                          std::string task_name,
                          const bool temporary_variable ){
    if ( !temporary_variable ){
      register_variable_work( name, dep, nGhost, dw, var_reg, time_substep, task_name );
    }
  }

  void register_variable( std::string name,
                          ArchesFieldContainer::VAR_DEPEND dep,
                          int nGhost,
                          ArchesFieldContainer::WHICH_DW dw,
                          std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                          std::string task_name,
                          const bool temporary_variable ){
    if ( !temporary_variable ){
      register_variable_work( name, dep, nGhost, dw, var_reg, 0, task_name );
    }
  }

  void register_variable( std::string name,
                          ArchesFieldContainer::VAR_DEPEND dep,
                          std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                          std::string task_name,
                          const bool temporary_variable ){
    ArchesFieldContainer::WHICH_DW dw = ArchesFieldContainer::NEWDW;
    int nGhost = 0;
    if ( !temporary_variable ){
      register_variable_work( name, dep, nGhost, dw, var_reg, 0, task_name );
    }
  }

  void register_variable( std::string name,
                          ArchesFieldContainer::VAR_DEPEND dep,
                          std::vector<ArchesFieldContainer::VariableInformation>& var_reg,
                          const int timesubstep,
                          std::string task_name,
                          const bool temporary_variable ){

    ArchesFieldContainer::WHICH_DW dw = ArchesFieldContainer::NEWDW;
    int nGhost = 0;
    if ( !temporary_variable ){
      register_variable_work( name, dep, nGhost, dw, var_reg, timesubstep, task_name );
    }
  }
} // namespace Uintah
