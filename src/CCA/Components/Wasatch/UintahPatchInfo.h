#ifndef Expr_UintahPatchInfo
#define Expr_UintahPatchInfo

namespace Expr{

  typedef std::map< int, SpatialOps::OperatorDatabase* > PatchOperators;

  struct UintahPatchInfo{
    PatchOperators patchOps; // one db per patch
    Uintah::Material* material,
  };

}


#endif // Expr_UintahPatchInfo
