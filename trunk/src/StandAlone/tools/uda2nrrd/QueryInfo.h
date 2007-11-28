#ifndef UDA2NRRD_QUERY_INFO_H
#define UDA2NRRD_QUERY_INFO_H

#include <Core/Util/ConsecutiveRangeSet.h>

#include <Core/DataArchive/DataArchive.h>

using namespace Uintah;

class QueryInfo {
public:
  QueryInfo() {}
  QueryInfo( DataArchive* archive,
             GridP grid,
             LevelP level,
             string varname,
             ConsecutiveRangeSet materials,
             int timestep,
             bool combine_levels,
             const Uintah::TypeDescription *type ) :
    archive(archive), grid(grid),
    level(level), varname(varname),
    materials(materials), timestep(timestep),
    combine_levels(combine_levels),
    type(type)
  {}
  
  DataArchive* archive;
  GridP grid;
  LevelP level;
  string varname;
  ConsecutiveRangeSet materials;
  int timestep;
  bool combine_levels;
  const Uintah::TypeDescription *type;
};

#endif // UDA2NRRD_QUERY_INFO_H
