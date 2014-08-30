get_filename_component( dir ${CMAKE_CURRENT_LIST_FILE} PATH )

if( ENABLE_UINTAH )
  add_subdirectory( uintah )
endif( ENABLE_UINTAH )

if( ENABLE_LBMS )
  add_subdirectory( lbms )
endif( ENABLE_LBMS )

set( src
  ${dir}/ExpressionBase.cpp
  ${dir}/ExpressionFactory.cpp
  ${dir}/ExpressionTree.cpp
  ${dir}/Context.cpp
  ${dir}/FieldDeps.cpp
  ${dir}/FieldManagerBase.cpp
  ${dir}/FieldManagerList.cpp
  ${dir}/HybridScheduler.cpp
  ${dir}/Poller.cpp
  ${dir}/PriorityScheduler.cpp
  ${dir}/SchedulerBase.cpp
  ${dir}/Tag.cpp
  ${dir}/TransportEquation.cpp
  ${dir}/VertexProperty.cpp
  )

set( expr_h
  ExprFwd.h
  ExprDeps.h
  Context.h
  Expression.h
  ExpressionID.h
  ExpressionBase.h
  ExpressionFactory.h
  ExpressionTree.h
  FieldDeps.h
  FieldManagerBase.h
  FieldManagerList.h
  Functions.h
  GraphType.h
  HybridScheduler.h
  ManagerTypes.h
  Poller.h
  PriorityScheduler.h
  PropertyStash.h
  Schedulers.h
  SchedulerBase.h
  SourceExprOp.h
  Tag.h
  TransportEquation.h
  VertexProperty.h
)

if( NOT ENABLE_UINTAH )
  set( expr_h ${expr_h} TimeStepper.h )
  if( ENABLE_OUTPUT AND NOT ENABLE_LBMS )
    set( src ${src} ${dir}/FieldWriter.cpp )
    set( expr_h ${expr_h} ${dir}/FieldWriter.h )
  endif()
endif()

if( DefaultFieldManager OR ENABLE_LBMS)
  set( expr_h ${expr_h} DefaultFieldManager.h ExprPatch.h )
endif()
if( DefaultFieldManager )
  set( src ${src}
    ${dir}/ExprPatch.cpp
    )
endif()

# INSTALLATION
#install( TARGETS exprlib ARCHIVE DESTINATION lib/expression )

install( FILES ${expr_h}
  DESTINATION include/expression
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )

set( expr_sources ${expr_sources}
  ${src}
  PARENT_SCOPE
  )
