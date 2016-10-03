#ifndef Uintah_Component_Arches_BOUNDARYFUNCTORS_h
#define Uintah_Component_Arches_BOUNDARYFUNCTORS_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Task/FieldContainer.h>

namespace Uintah { namespace BoundaryFunctor{

  // Here are the instructions for adding a new BC functor:
  // 1) Define an empty struct for the BC type (e.g., struct SwirlType)
  // 2) Define a templated BCFunctor using the empty struct from 1)
  // 3) Add an enum to the BC_FUNCTORS definition (append it to the end)
  // 4) Add the new BCFunctor to the tuple (BCFunctorStorageType) at the end of this file.

  //NOTICE:
  // The ordering of the BCFunctorType below must match the enum ordering here
  enum BC_FUNCTORS
  {
    SWIRL,
    FILE,
    MASSFLOW,
    STABLE,
    TABLELOOKUP
  };

  struct SwirlType{};
  struct FileType{};
  struct MassFlowType{};
  struct StableType{};
  struct TableLookupType{};

  //static const Uintah::ProblemSpecP get_uintah_bc_problem_spec( const Uintah::ProblemSpecP& db, std::string name, std::string face, std::string type ){
    //if ( db->findBlock("BoundaryConditions") ){
      //Uintah::ProblemSpecP db_bc = db->findBlock("BoundaryConditions");
      //for ( Uintah::ProblemSpecP db_var_bc = db_bc->findBlock("bc"); db_var_bc != 0;
            //db_var_bc = db_var_bc->findNextBlock("bc") ){

        //std::string att_name, att_face, att_type;
        //db_var_bc->getAttribute("varname", att_name);
        //db_var_bc->getAttribute("face", att_face);
        //db_var_bc->getAttribute("type", att_type);

        //if ( att_name == name && att_face == face && att_type == type ){
          //return db_var_bc;
        //}
      //}
      //std::stringstream msg;
      //msg << "Error: Cannot find a matching bc in <ARCHES><BoundaryConditions> for [varname, face, type]: [" << name << ", " << face << ", " << type << "]" << std::endl;
      //throw Uintah::ProblemSetupException(msg.str(), __FILE__, __LINE__);
    //} else {
      //throw Uintah::ProblemSetupException("Error: Cannot find <ARCHES><BoundaryConditions input block.", __FILE__, __LINE__ );
    //}
  //}

  // static const Uintah::ProblemSpecP get_face_spec( const Uintah::ProblemSpecP& db, std::string face_name ){
  //
  //   Uintah::ProblemSpecP db_bc_root = db->getRootNode()->findBlock("Grid")->findBlock("BoundaryConditions");
  //
  //   for (Uintah::ProblemSpecP db_face = db_bc_root->findBlock("Face"); db_face != 0; db_face = db_face->findNextBlock("Face") ){
  //     std::string this_face_name = "NOTNAMED";
  //     db_face->getAttribute("name", this_face_name);
  //     if ( this_face_name != "NOTNAMED" ){
  //       if ( this_face_name == face_name ){
  //         return db_face;
  //       }
  //     } else {
  //       throw Uintah::ProblemSetupException("Error: You are using a boundary feature that requires that all <Face>'s have a unique name attribute specificed",__FILE__,__LINE__);
  //     }
  //   }
  //   throw Uintah::ProblemSetupException("Error: Cannot locate the face named: "+face_name, __FILE__, __LINE__);
  // }

/**
  <BoundaryConditions>
    <bc varname="a" face="circle-x-minus" type="swirl">
      ...inputs...
    </bc>
  </BoundaryConditions
**/

  // GENERIC FUNCTOR -------------------------------------------------------------------------------
  template <typename T, BC_FUNCTORS>
  struct BCFunctor{
    BCFunctor(){
        throw InvalidValue("Error: You have hit the generic BC Functor.",__FILE__,__LINE__);
    }
    void problemSetup( const Uintah::ProblemSpecP& db){}
    void register_depenendencies( std::vector<Uintah::ArchesFieldContainer::VariableInformation>& reg ){}
    void eval_bc( const Patch* patch, TaskInterface::ArchesTaskInfoManager* tsk_info, T& var ){}
  };

  template <typename T>
  struct BCFunctor<T, SWIRL>{
    // BCFunctor(){}
    // void problemSetup( const Uintah::ProblemSpecP& db){}
    // void register_depenendencies( std::vector<Uintah::ArchesFieldContainer::VariableInformation>& reg ){}
    // void eval_bc( const Patch* patch, TaskInterface::ArchesTaskInfoManager* tsk_info, T& var ){}
  };

  template <typename T>
  struct BCFunctor<T, FILE>{
    // BCFunctor(){}
    // void problemSetup( const Uintah::ProblemSpecP& db){}
    // void register_depenendencies( std::vector<Uintah::ArchesFieldContainer::VariableInformation>& reg ){}
    // void eval_bc( const Patch* patch, TaskInterface::ArchesTaskInfoManager* tsk_info, T& var ){}
  };

  //Useful typedefs:
  typedef BCFunctor<CCVariable<double>, BoundaryFunctor::SWIRL> CCFunSwirlStore;
  typedef BCFunctor<CCVariable<double>, BoundaryFunctor::FILE> CCFunFileStore;

  static std::tuple< std::vector<CCFunSwirlStore>,
                     std::vector<CCFunFileStore> > CCBCFunctorStorage;

    //EXAMPLE USING THE STORAGE MECHANISM
    // using namespace BoundaryFunctor;
    //
    // CCFunSwirlStore test1;
    // CCFunFileStore test2;
    //
    // const int i = SWIRL;
    //
    // std::vector<CCFunSwirlStore>& my_ptr = std::get<i>(CCBCFunctorStorage);
    // my_ptr.push_back(test1);
    //
    // std::vector<CCFunFileStore>& my_ref2 = std::get<1>(CCBCFunctorStorage);
    // my_ref2.push_back(test2);
    // END EXAMPLE


  // // -----------------------------------------------------------------------------------------------
  // struct Swirl : BCFunctorBase {
  //
  //   Swirl( const Uintah::ProblemSpecP& db, const std::string name, const std::string face,
  //          const std::string type, double density){
  //
  //     problemSetup(db, name, face, type);
  //
  //     m_density = density;
  //
  //   }
  //   void problemSetup( const Uintah::ProblemSpecP& db, const std::string name,
  //                      const std::string face, const std::string type ){
  //
  //     const Uintah::ProblemSpecP& db_bc = get_uintah_bc_problem_spec( db, name, face, type );
  //
  //     // double m_swirl_no, m_flowrate;
  //     // Uintah::Vector m_centroid;
  //     // db_bc->require("swirl_number", m_swirl_no);
  //     // db_dc->require("centroid", m_centroid);
  //     // db_dc->require("flowrate", m_flowrate);
  //
  //   }
  //
  //
  // private:
  //
  //   double m_swirl_no;
  //   double m_flowrate;
  //   double m_density;
  //   Uintah::Vector m_centroid;
  //
  // };

  // // -----------------------------------------------------------------------------------------------
  // template<>
  // struct BCFunctor<FileType>{
  //   BCFunctor(){
  //   }
  //
  //   void operator()(int i, int j, int k){
  //   }
  // };
  //
  // // -----------------------------------------------------------------------------------------------
  // template<>
  // struct BCFunctor<MassFlowType>{
  //   BCFunctor(){
  //   }
  //
  //   void operator()(int i, int j, int k){
  //   }
  // };
  //
  // // -----------------------------------------------------------------------------------------------
  // template<>
  // struct BCFunctor<StableType>{
  //   BCFunctor(){
  //   }
  //
  //   void operator()(int i, int j, int k){
  //   }
  // };
  //
  // // -----------------------------------------------------------------------------------------------
  // template<>
  // struct BCFunctor<TableLookupType, MixingRxnModel>{
  //   BCFunctor( MixingRxnModel* mix_table ){
  //   }
  //
  //   void operator()(int i, int j, int k){
  //   }
  // };

  // -----------------------------------------------------------------------------------------------
  // typedef std::tuple<
  //             std::vector<BCFunctor<SwirlType> >,
  //             std::vector<BCFunctor<FileType> >,
  //             std::vector<BCFunctor<MassFlowType> >,
  //             std::vector<BCFunctor<StableType> >,
  //             std::vector<BCFunctor<TableLookupType, MixingRxnModel> >
  //           > BCFunctorStorageType;

  // typedef std::map<std::string, BCFunctorBase > BCFunctorStorageType;
  // typedef std::map<int, BCFunctorStorageType > PatchToBCFunctorStorage;

} } //end namespace Uintah::BoundaryFunctor


#endif
