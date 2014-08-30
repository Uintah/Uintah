#include <spatialops/SpatialOpsTools.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cmath>

#include <spatialops/OperatorDatabase.h>

#include <spatialops/Nebo.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialMask.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FieldComparisons.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/stencil/FVStaggeredBCOp.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>
#include <spatialops/structured/FVStaggeredBCTools.h>
#include <spatialops/structured/stencil/StencilBuilder.h>

#include <test/TestHelper.h>
#include <spatialops/structured/FieldHelper.h>

using namespace SpatialOps;

int main()
{
  TestHelper status(true);

  typedef SVolField PhiFieldT;
  typedef SSurfXField GammaFieldT;

  const IntVec dim(3, 3, 1);
  const double length = 1.5;
  const double dx = length/dim[0];
  const double dy = length/dim[1];
  const std::vector<bool> bcFlag(3,true);
  const GhostData  fghost(1);
  const GhostData dfghost(1);

  {
    const BoundaryCellInfo  fbc = BoundaryCellInfo::build<PhiFieldT>( bcFlag[0], bcFlag[1], bcFlag[2] );
    const BoundaryCellInfo dfbc = BoundaryCellInfo::build<GammaFieldT>( bcFlag[0], bcFlag[1], bcFlag[2] );
    SpatFldPtr<PhiFieldT  > test  = SpatialFieldStore::get_from_window<PhiFieldT>( get_window_with_ghost(dim, fghost, fbc),  fbc,  fghost );
    SpatFldPtr<PhiFieldT  > ref   = SpatialFieldStore::get_from_window<PhiFieldT>( get_window_with_ghost(dim, fghost, fbc),  fbc,  fghost );
    SpatFldPtr<GammaFieldT> gamma = SpatialFieldStore::get_from_window<GammaFieldT>( get_window_with_ghost(dim,dfghost,dfbc), dfbc, dfghost );

    /* Invert gradent:
     *  Cell (volume) to face:
     *  -------------
     *  |     |     |
     *  |  A  B  C  |
     *  |     |     |
     *  -------------
     *
     *  dx = Lx/nx
     *
     *  Gradient:
     *  B = A * -1/dx + C * 1/dx
     *
     *  Negative side inversion:
     *  A = (B - C * 1/dx) / (-1/dx)
     *
     *  Positive side inversion:
     *  C = (B - A * -1/dx) / (1/dx)
     *
     *  Indices:
     *  A = -1, 0, 0
     *  B =  0, 0, 0
     *  C =  0, 0, 0
     */

    GammaFieldT::iterator ig = gamma->begin();
    PhiFieldT::iterator ir = ref->begin();
    PhiFieldT::iterator it = test->begin();
    for(int j=-1; j < 4; j++)
      for( int i = -1; i < 4; i++) {
        *it = i + j * 5;
        if( (i == -1 && j == 1) ||
            (i ==  0 && j == 2) ) {
          //Negative side inversion: *ir == A
          int B = 25 + (i + 1) + j * 5;
          int C = (i + 1) + j * 5;
          *ir = (B - C * (1/dx)) / (-1/dx);
        }
        else if( (i ==  2 && j == 1) ||
                 (i ==  3 && j == 2) ) {
          //Positive side inversion: *ir == C
          int B = 25 + (i) + j * 5;
          int A = (i - 1) + j * 5;
          *ir = (B - A * (-1/dx)) / (1/dx);
        }
        else
          *ir = i + j * 5;
        it++;
        ir++;
      }
    for(int j=-1; j < 4; j++)
      for( int i = -1; i < 5; i++) {
        *ig = 25 + i + j * 5;
        ig++;
      }

    print_field(*gamma,std::cout);
    //make the BC:
    OperatorDatabase opdb;
    build_stencils( dim[0], dim[1], dim[2], length, length, length, opdb );
    typedef BasicOpTypes<PhiFieldT>::GradX OpT;
    const OpT* const op = opdb.retrieve_operator<OpT>();
    NeboBoundaryConditionBuilder<OpT> BC(*op);

    //make the minus mask:
    std::vector<IntVec> minusSet;
    minusSet.push_back(IntVec(0,1,0));
    minusSet.push_back(IntVec(1,2,0));
    SpatialMask<GammaFieldT> minus(*gamma, minusSet);

    // evaluate the minus BC and set it in the field.
    BC(minus, *test, *gamma, true);

    //make the plus mask:
    std::vector<IntVec> plusSet;
    plusSet.push_back(IntVec(2,1,0));
    plusSet.push_back(IntVec(3,2,0));
    //build mask without gamma:
    SpatialMask<GammaFieldT> plus = SpatialMask<GammaFieldT>::build(*test, plusSet);

    // evaluate the plus BC and set it in the field.
    BC(plus, *test, *gamma, false);

    //display differences and values:
    display_fields_compare(*test, *ref, true, true);

    // verify that the BC was set properly.
    status( field_equal(*test, *ref), "Invert GradX on SVol" );
  }

  {
    const BoundaryCellInfo  fbc = BoundaryCellInfo::build<XVolField  >( bcFlag[0], bcFlag[1], bcFlag[2] );
    const BoundaryCellInfo dfbc = BoundaryCellInfo::build<XSurfXField>( bcFlag[0], bcFlag[1], bcFlag[2] );
    SpatFldPtr<XVolField  > test  = SpatialFieldStore::get_from_window<XVolField>( get_window_with_ghost(dim, fghost, fbc),  fbc,  fghost );
    SpatFldPtr<XVolField  > ref   = SpatialFieldStore::get_from_window<XVolField>( get_window_with_ghost(dim, fghost, fbc),  fbc,  fghost );
    SpatFldPtr<XSurfXField> gamma = SpatialFieldStore::get_from_window<XSurfXField>( get_window_with_ghost(dim,dfghost,dfbc), dfbc, dfghost );

    /* Invert gradent:
     *  X-staggered Cell (volume) to face:
     *  -------
     *  |     |
     *  X  S  X
     *  |     |
     *  -------
     *  X => SSurfX AND XVol
     *  S => SVol   AND XSurfX
     *
     *  -------
     *  |     |
     *  A  B  C
     *  |     |
     *  -------
     *
     *  dx = Lx/nx
     *
     *  Gradient:
     *  B = A * -1/dx + C * 1/dx
     *
     *  Negative side inversion:
     *  A = (B - C * 1/dx) / (-1/dx)
     *
     *  Positive side inversion:
     *  C = (B - A * -1/dx) / (1/dx)
     *
     *  Indices:
     *  A =  0, 0, 0
     *  B =  0, 0, 0
     *  C =  1, 0, 0
     */

    XSurfXField::iterator ig = gamma->begin();
    XVolField::iterator ir = ref->begin();
    XVolField::iterator it = test->begin();
    for(int j=-1; j < 4; j++)
      for( int i = -1; i < 4; i++) {
        *ig = 25 + i + j * 5;
        ig++;
      }
    for(int j=-1; j < 4; j++)
      for( int i = -1; i < 5; i++) {
        *it = i + j * 5;
        if( (i == -1 && j == 1) ||
            (i == -1 && j == 2) ) {
          //Negative side inversion: *ir == A
          int B = 25 + (i) + j * 5;
          int C = (i + 1) + j * 5;
          *ir = (B - C * (1/dx)) / (-1/dx);
        }
        else if( (i ==  4 && j == 1) ||
                 (i ==  4 && j == 2) ) {
          //Positive side inversion: *ir == C
          int B = 25 + (i - 1) + j * 5;
          int A = (i - 1) + j * 5;
          *ir = (B - A * (-1/dx)) / (1/dx);
        }
        else
          *ir = i + j * 5;
        it++;
        ir++;
      }

    print_field(*gamma,std::cout);
    //make the BC:
    OperatorDatabase opdb;
    build_stencils( dim[0], dim[1], dim[2], length, length, length, opdb );
    typedef BasicOpTypes<XVolField>::GradX OpT;
    const OpT* const op = opdb.retrieve_operator<OpT>();
    NeboBoundaryConditionBuilder<OpT> BC(*op);

    //make the minus mask:
    std::vector<IntVec> minusSet;
    minusSet.push_back(IntVec(-1,1,0));
    minusSet.push_back(IntVec(-1,2,0));
    SpatialMask<XSurfXField> minus(*gamma, minusSet);

    // evaluate the minus BC and set it in the field.
    BC(minus, *test, *gamma, true);

    //make the plus mask:
    std::vector<IntVec> plusSet;
    plusSet.push_back(IntVec(3,1,0));
    plusSet.push_back(IntVec(3,2,0)); //??
    //build mask without gamma:
    SpatialMask<XSurfXField> plus = SpatialMask<XSurfXField>::build(*test, plusSet);
    std::cout << "plus: " << plus.window_with_ghost().extent() << std::endl;
    // evaluate the plus BC and set it in the field.
    BC(plus, *test, *gamma, false);

    //display differences and values:
    display_fields_compare(*test, *ref, true, true);

    // verify that the BC was set properly.
    status( field_equal(*test, *ref), "Invert GradX on XVol" );
  }

  {
    const BoundaryCellInfo  fbc = BoundaryCellInfo::build<YVolField  >( bcFlag[0], bcFlag[1], bcFlag[2] );
    const BoundaryCellInfo dfbc = BoundaryCellInfo::build<YSurfYField>( bcFlag[0], bcFlag[1], bcFlag[2] );
    SpatFldPtr<YVolField  > test  = SpatialFieldStore::get_from_window<YVolField>( get_window_with_ghost(dim, fghost, fbc),  fbc,  fghost );
    SpatFldPtr<YVolField  > ref   = SpatialFieldStore::get_from_window<YVolField>( get_window_with_ghost(dim, fghost, fbc),  fbc,  fghost );
    SpatFldPtr<YSurfYField> gamma = SpatialFieldStore::get_from_window<YSurfYField>( get_window_with_ghost(dim,dfghost,dfbc), dfbc, dfghost );

    /* Invert gradent:
     *  Y-staggered Cell (volume) to face:
     *  ---Y---
     *  |     |
     *  |  S  |
     *  |     |
     *  ---Y---
     *  Y => SSurfY AND YVol
     *  S => SVol   AND YSurfY
     *
     *  ---C---
     *  |     |
     *  |  B  |
     *  |     |
     *  ---A---
     *
     *  dy = Ly/ny
     *
     *  Gradient:
     *  B = A * -1/dy + C * 1/dy
     *
     *  Negative side inversion:
     *  A = (B - C * 1/dy) / (-1/dy)
     *
     *  Positive side inversion:
     *  C = (B - A * -1/dy) / (1/dy)
     *
     *  Indices:
     *  A =  0, 0, 0
     *  B =  0, 0, 0
     *  C =  0, 1, 0
     */

    YSurfYField::iterator ig = gamma->begin();
    YVolField::iterator ir = ref->begin();
    YVolField::iterator it = test->begin();
    for(int j=-1; j < 4; j++)
      for( int i = -1; i < 4; i++) {
        *ig = 25 + i + j * 5;
        ig++;
      }
    for(int j=-1; j < 5; j++)
      for( int i = -1; i < 4; i++) {
        *it = i + j * 5;
        if( (i == 0 && j == -1) ||
            (i == 1 && j == -1) ) {
          //Negative side inversion: *ir == A
          int B = 25 + i + (j) * 5;
          int C = i + (j + 1) * 5;
          *ir = (B - C * (1/dy)) / (-1/dy);
        }
        else if( (i ==  0 && j == 4) ||
                 (i ==  3 && j == 4) ) {
          //Positive side inversion: *ir == C
          int B = 25 + i + (j - 1) * 5;
          int A = i + (j - 1) * 5;
          *ir = (B - A * (-1/dy)) / (1/dy);
        }
        else
          *ir = i + j * 5;
        it++;
        ir++;
      }

    print_field(*gamma,std::cout);
    //make the BC:
    OperatorDatabase opdb;
    build_stencils( dim[0], dim[1], dim[2], length, length, length, opdb );
    typedef BasicOpTypes<YVolField>::GradY OpT;
    const OpT* const op = opdb.retrieve_operator<OpT>();
    NeboBoundaryConditionBuilder<OpT> BC(*op);

    //make the minus mask:
    std::vector<IntVec> minusSet;
    minusSet.push_back(IntVec(0,-1,0));
    minusSet.push_back(IntVec(1,-1,0));
    SpatialMask<YSurfYField> minus(*gamma, minusSet);

    // evaluate the minus BC and set it in the field.
    BC(minus, *test, *gamma, true);

    //make the plus mask:
    std::vector<IntVec> plusSet;
    plusSet.push_back(IntVec(0,3,0));
    plusSet.push_back(IntVec(3,3,0));
    //build mask without gamma:
    SpatialMask<YSurfYField> plus = SpatialMask<YSurfYField>::build(*test, plusSet);

    // evaluate the plus BC and set it in the field.
    BC(plus, *test, *gamma, false);

    //display differences and values:
    display_fields_compare(*test, *ref, true, true);

    // verify that the BC was set properly.
    status( field_equal(*test, *ref), "Invert GradY on YVol" );
  }

  {
    const BoundaryCellInfo  fbc = BoundaryCellInfo::build<XVolField  >( bcFlag[0], bcFlag[1], bcFlag[2] );
    const BoundaryCellInfo dfbc = BoundaryCellInfo::build<XVolField>( bcFlag[0], bcFlag[1], bcFlag[2] );
    SpatFldPtr<XVolField  > test  = SpatialFieldStore::get_from_window<XVolField>( get_window_with_ghost(dim, fghost, fbc),  fbc,  fghost );
    SpatFldPtr<XVolField  > ref   = SpatialFieldStore::get_from_window<XVolField>( get_window_with_ghost(dim, fghost, fbc),  fbc,  fghost );
    SpatFldPtr<XVolField> gamma = SpatialFieldStore::get_from_window<XVolField>( get_window_with_ghost(dim,dfghost,dfbc), dfbc, dfghost );

    /* Invert gradent:
     *  Y-staggered Cell (volume) to face:
     *  -------------
     *  |     |     |
     *  X  S  X  S  X
     *  |     |     |
     *  -------------
     *  X => XVol
     *  S => SVol
     *
     *  -------------
     *  |     |     |
     *  A     B     C
     *  |     |     |
     *  -------------
     *
     *  dx = Lx/nx
     *
     *  GradientX (FDStencil):
     *  B = A * -0.5/dx + C * 0.5/dx
     *
     *  Negative side inversion:
     *  A = (B - C * 0.5/dx) / (-0.5/dx)
     *
     *  Positive side inversion:
     *  C = (B - A * -0.5/dx) / (0.5/dx)
     *
     *  Indices:
     *  A = -1, 0, 0
     *  B =  0, 0, 0
     *  C =  1, 0, 0
     */

    XVolField::iterator ig = gamma->begin();
    XVolField::iterator ir = ref->begin();
    XVolField::iterator it = test->begin();
    for(int j=-1; j < 4; j++)
      for( int i = -1; i < 5; i++) {
        *ig = 25 + i + j * 5;
        *it = i + j * 5;
        if( (i == -1 && j == 0) ||
            (i == -1 && j == 1) ) {
          //Negative side inversion: *ir == A
          int B = 25 + (i + 1) + j * 5;
          int C = (i + 2) + j * 5;
          *ir = (B - C * (0.5/dx)) / (-0.5/dx);
        }
        else if( (i ==  4 && j == 0) ||
                 (i ==  4 && j == 1) ) {
          //Positive side inversion: *ir == C
          int B = 25 + (i - 1) + j * 5;
          int A = (i - 2) + j * 5;
          *ir = (B - A * (-0.5/dx)) / (0.5/dx);
        }
        else
          *ir = i + j * 5;
        it++;
        ir++;
        ig++;
      }

    print_field(*gamma,std::cout);
    //make the BC:
    OperatorDatabase opdb;
    build_stencils( dim[0], dim[1], dim[2], length, length, length, opdb );
    typedef OperatorTypeBuilder<GradientX,XVolField,XVolField>::type OpT;
    const OpT* const op = opdb.retrieve_operator<OpT>();
    NeboBoundaryConditionBuilder<OpT> BC(*op);

    //make the minus mask:
    std::vector<IntVec> minusSet;
    minusSet.push_back(IntVec(0,0,0));
    minusSet.push_back(IntVec(0,1,0));
    SpatialMask<XVolField> minus(*gamma, minusSet);

    // evaluate the minus BC and set it in the field.
    BC(minus, *test, *gamma, true);

    //make the plus mask:
    std::vector<IntVec> plusSet;
    plusSet.push_back(IntVec(3,0,0));
    plusSet.push_back(IntVec(3,1,0));
    //build mask without gamma:
    SpatialMask<XVolField> plus = SpatialMask<XVolField>::build(*test, plusSet);

    // evaluate the plus BC and set it in the field.
    BC(plus, *test, *gamma, false);

    //display differences and values:
    display_fields_compare(*test, *ref, true, true);

    // verify that the BC was set properly.
    status( field_equal(*test, *ref), "Invert GradX(FD) on XVol" );
  }


  if( status.ok() ) {
    std::cout << "ALL TESTS PASSED :)" << std::endl;
    return 0;
  } else {
    std::cout << "******************************" << std::endl
              << " At least one test FAILED! :(" << std::endl
              << "******************************" << std::endl;
    return -1;
  }

}
