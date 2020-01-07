#ifndef Uintah_Component_Arches_PressureBC_h
#define Uintah_Component_Arches_PressureBC_h

#include <CCA/Components/Arches/Task/AtomicTaskInterface.h>

//===============================================================

/**
* @class  PressureBC
* @author Jeremy Thornock
* @date   2017
*
* @brief Sets the boundary condition on the pressure so that the
*        the velocity update (u_new = u_hat - grad P) can be
*        uniformly applied.
*
**/

//===============================================================

namespace Uintah{ namespace ArchesCore{

  class PressureBC : AtomicTaskInterface {

public:

    /** @brief Default constructor **/
    PressureBC( std::string task_name, int matl_index );

    /** @brief Default destructor **/
    ~PressureBC();

    /** @brief Input file interface **/
    void problemSetup( ProblemSpecP& db );

    /** @brief Create local labels for the task **/
    void create_local_labels();

    /** @brief Registers all variables with pertinent information for the
     *         uintah dw interface **/
    void register_timestep_eval(
      std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
      const int time_substep, const bool pack_tasks );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    /** @brief Builder class containing instructions on how to build the task **/
    class Builder : public AtomicTaskInterface::AtomicTaskBuilder {

      public:

        Builder(std::string name, int matl_index):
        m_task_name(name), m_matl_index(matl_index){};

        ~Builder() {}

        PressureBC* build(){ return scinew PressureBC(m_task_name, m_matl_index);};

      protected:

        std::string m_task_name;
        int m_matl_index;

    };

private:

    std::string m_press;

  };
} } // namespace Uintah::ArchesCore

#endif
