# threadpool is a header - only library

get_filename_component( dir ${CMAKE_CURRENT_LIST_FILE} PATH )

install( FILES ${dir}/threadpool.hpp
  DESTINATION include/spatialops/threadpool
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ 
  )

install( DIRECTORY ${dir}/threadpool
  DESTINATION include/spatialops/threadpool
  PATTERN ".svn" EXCLUDE
  )
