/*
 * DeviceTypes.h
 *
 */

#ifndef MEMORYTYPES_H_
#define MEMORYTYPES_H_
#include <string>
#include <sstream>
#include <stdexcept>

namespace SpatialOps{

#define CPU_INDEX -1
#define GPU_INDEX  0

#define IS_CPU_INDEX(INDEX)   (INDEX == CPU_INDEX)
#define IS_GPU_INDEX(INDEX)   (INDEX >= GPU_INDEX)
#define IS_VALID_INDEX(INDEX) (INDEX >= CPU_INDEX)

namespace DeviceTypeTools {
  /**
   *  \class DeviceTypeTools
   *  \brief Provide descriptions
   */
  inline std::string get_memory_type_description( short int deviceIndex ) {
    if( deviceIndex == CPU_INDEX ){
      return std::string("(Locally allocated, generic system RAM)");
    }
    else if( IS_GPU_INDEX(deviceIndex) ){
      return std::string("(Externally allocated, CUDA GPU device)");
    }
    else{
      return std::string("(Unknown or Invalid)");
    }
  }

  /**
   *  \class DeviceTypeTools
   *  \brief Throw an exception if given index is not valid
   */
  inline void check_valid_index( short int deviceIndex, const char* file, int line) {
    if( !IS_VALID_INDEX(deviceIndex) ) {
      std::ostringstream msg;
      msg << "Given an unknown device index. Given: " << deviceIndex << " ("
          << DeviceTypeTools::get_memory_type_description(deviceIndex) << ")" << std::endl
          << "\t - " << file << " : " << line << std::endl;
      throw(std::runtime_error(msg.str()));
    }
  }

} // namespace DeviceTypeTools
} // namespace SpatialOps

#endif /* DEVICETYPES_H_ */
