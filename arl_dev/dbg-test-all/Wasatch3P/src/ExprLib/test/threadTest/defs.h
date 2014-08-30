#ifndef defs_h
#define defs_h

#include <spatialops/structured/FVStaggered.h>

#include <expression/ExprPatch.h>

typedef SpatialOps::SVolField      VolT;
typedef SpatialOps::SSurfXField    XFluxT;

typedef SpatialOps::BasicOpTypes<VolT>  OpTypes;
typedef OpTypes::DivX   XDivT;
typedef OpTypes::GradX  XGradT;

typedef Expr::ExprPatch                        PatchT;

#endif
