#ifndef Uintah_Components_Arches_HandoffHelper_h
#define Uintah_Components_Arches_HandoffHelper_h
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>

#include <map>

namespace Uintah { namespace ArchesCore{

class HandoffHelper{

public:

  HandoffHelper(){}
  ~HandoffHelper(){}

  typedef std::map<IntVector, double> CellToValue;
  typedef std::map<IntVector, Vector> CellToVector;

  /** @brief From file info container **/
  struct FFInfo {
    CellToValue values;
    CellToVector vec_values; 
    Vector relative_xyz;
    double dx;
    double dy;
    IntVector relative_ijk;
    std::string default_type;
    std::string name;
    double default_value;
  };

  /** @brief Read information from an input file and return the information in a container **/
  /** If the input file doesn't have vector information, pass index=-1 **/
  void readInputFile( std::string file_name, const int index, FFInfo& info );

  /** @brief Read information from an input file and return the information in a container **/
  /** Note that this version assumes that the file has three vector components which it reads **/
  void readInputFile( std::string file_name, FFInfo& info );

private:

};

}} //namespace Uintah::ArchesCore




#endif
