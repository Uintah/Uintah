#include <spatialops/structured/FVStaggeredFieldTypes.h>

#include <spatialops/Nebo.h>

#include <iostream>
#include <vector>

using namespace SpatialOps;

//The goal of this file is to test that all possible combinations of clauses compile.
//Correctness of results is not part of this file.
//Currently, correctness is tested in test/NeboTest.cpp

int main()
{
  const int nx=10, ny=12, nz=14;

  typedef SVolField Field;
  typedef SingleValueField SVField;
  
  const GhostData oneGhost(1);
  const GhostData noGhost(0);
  const BoundaryCellInfo   bc = BoundaryCellInfo::build<Field>(true,true,true);
  const BoundaryCellInfo svbc = BoundaryCellInfo::build<SVField>(true,true,true);
  const MemoryWindow   window( get_window_with_ghost(IntVec(nx,ny,nz),oneGhost,bc) );
  const MemoryWindow svwindow( get_window_with_ghost(IntVec(1,1,1),oneGhost,svbc) );


  Field ffa( window, bc, oneGhost, NULL );
  Field ffb( window, bc, oneGhost, NULL );
  SVField sva( svwindow, svbc, noGhost, NULL );
  SVField svb( svwindow, svbc, noGhost, NULL );
  double dtest;
  
  std::vector<Field> vec = std::vector<Field>();

  int const max = nx * ny * nz;

  Field::iterator ia = ffa.begin();
  for(int i = 1; ia!=ffa.end(); i++, ++ia) {
    *ia = i;
  }
  sva[0] = max;

  std::vector<IntVec> points(0);
  SpatialMask<Field> mask(ffa, points);

  // Testing here is broken down into the different cases:

  // Previous clauses can be: None, Simple, Single Value, or Full
  // Condition expressions can be: Final, Boolean, Single Value Expression, Full Expression, or Mask
  // Value expressions can be: Double Single Value Field, Single Value Expression, Full Field, or Full Expression

  // Simple refers to using only scalar values
  // Single Value refers to using Single Value fields and scalar values only
  // Full refers to using at least one SpatialField that is not a Single Value Field

  ////////////////////////////////////////////////
  // Previous: None //////////////////////////////
  ////////////////////////////////////////////////

  ////// Condition: Final ////////////////////////

  //Value: Double
  ffb <<= cond(1.0);
  svb <<= cond(1.0);
  dtest = cond(1.0);

  //Value: Single Value Field
  ffb <<= cond(sva);
  svb <<= cond(sva);

  //Value: Single Value Expression
  ffb <<= cond(sva + sva);
  svb <<= cond(sva + sva);

  //Value: Full Field
  ffb <<= cond(ffa);

  //Value: Full Expression
  ffb <<= cond(ffa + ffa);

  ////// Condition: Boolean //////////////////////

  //Value: Double
  ffb <<= cond(false, 1.0)
              (2.0);
  svb <<= cond(false, 1.0)
              (2.0);
  dtest = cond(false, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(false, sva)
              (2.0);
  svb <<= cond(false, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(false, sva + sva)
              (2.0);
  svb <<= cond(false, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(false, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(false, ffa + ffa)
              (2.0);

  ////// Condition: Single Value Expression //////

  //Value: Double
  ffb <<= cond(sva < 0.0, 1.0)
              (2.0);
  svb <<= cond(sva < 0.0, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(sva < 0.0, sva)
              (2.0);
  svb <<= cond(sva < 0.0, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(sva < 0.0, sva + sva)
              (2.0);
  svb <<= cond(sva < 0.0, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(sva < 0.0, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(sva < 0.0, ffa + ffa)
              (2.0);

  ////// Condition: Full Expression //////////////

  //Value: Double
  ffb <<= cond(ffa < 0.0, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(ffa < 0.0, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(ffa < 0.0, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(ffa < 0.0, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(ffa < 0.0, ffa + ffa)
              (2.0);

  ////// Condition: Mask /////////////////////////

  //Value: Double
  ffb <<= cond(mask, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(mask, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(mask, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(mask, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(mask, ffa + ffa)
              (2.0);

  ////////////////////////////////////////////////
  // Previous: Simple ////////////////////////////
  ////////////////////////////////////////////////

  ////// Condition: Final ////////////////////////

  //Value: Double
  ffb <<= cond(false, 0.0)
              (1.0);
  svb <<= cond(false, 0.0)
              (1.0);
  dtest = cond(false, 0.0)
              (1.0);

  //Value: Single Value Field
  ffb <<= cond(false, 0.0)
              (sva);
  svb <<= cond(false, 0.0)
              (sva);

  //Value: Single Value Expression
  ffb <<= cond(false, 0.0)
              (sva + sva);
  svb <<= cond(false, 0.0)
              (sva + sva);

  //Value: Full Field
  ffb <<= cond(false, 0.0)
              (ffa);

  //Value: Full Expression
  ffb <<= cond(false, 0.0)
              (ffa + ffa);

  ////// Condition: Boolean //////////////////////

  //Value: Double
  ffb <<= cond(false, 0.0)
              (false, 1.0)
              (2.0);
  svb <<= cond(false, 0.0)
              (false, 1.0)
              (2.0);
  dtest = cond(false, 0.0)
              (false, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(false, 0.0)
              (false, sva)
              (2.0);
  svb <<= cond(false, 0.0)
              (false, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(false, 0.0)
              (false, sva + sva)
              (2.0);
  svb <<= cond(false, 0.0)
              (false, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(false, 0.0)
              (false, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(false, 0.0)
              (false, ffa + ffa)
              (2.0);

  ////// Condition: Single Value Expression //////

  //Value: Double
  ffb <<= cond(false, 0.0)
              (sva < 0.0, 1.0)
              (2.0);
  svb <<= cond(false, 0.0)
              (sva < 0.0, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(false, 0.0)
              (sva < 0.0, sva)
              (2.0);
  svb <<= cond(false, 0.0)
              (sva < 0.0, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(false, 0.0)
              (sva < 0.0, sva + sva)
              (2.0);
  svb <<= cond(false, 0.0)
              (sva < 0.0, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(false, 0.0)
              (sva < 0.0, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(false, 0.0)
              (sva < 0.0, ffa + ffa)
              (2.0);

  ////// Condition: Full Expression //////////////

  //Value: Double
  ffb <<= cond(false, 0.0)
              (ffa < 0.0, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(false, 0.0)
              (ffa < 0.0, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(false, 0.0)
              (ffa < 0.0, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(false, 0.0)
              (ffa < 0.0, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(false, 0.0)
              (ffa < 0.0, ffa + ffa)
              (2.0);

  ////// Condition: Mask /////////////////////////

  //Value: Double
  ffb <<= cond(false, 0.0)
              (mask, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(false, 0.0)
              (mask, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(false, 0.0)
              (mask, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(false, 0.0)
              (mask, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(false, 0.0)
              (mask, ffa + ffa)
              (2.0);

  ////////////////////////////////////////////////
  // Previous: Single Value //////////////////////
  ////////////////////////////////////////////////

  ////// Condition: Final ////////////////////////

  //Value: Double
  ffb <<= cond(sva < 0.0, 0.0)
              (1.0);
  svb <<= cond(sva < 0.0, 0.0)
              (1.0);

  //Value: Single Value Field
  ffb <<= cond(sva < 0.0, 0.0)
              (sva);
  svb <<= cond(sva < 0.0, 0.0)
              (sva);

  //Value: Single Value Expression
  ffb <<= cond(sva < 0.0, 0.0)
              (sva + sva);
  svb <<= cond(sva < 0.0, 0.0)
              (sva + sva);

  //Value: Full Field
  ffb <<= cond(sva < 0.0, 0.0)
              (ffa);

  //Value: Full Expression
  ffb <<= cond(sva < 0.0, 0.0)
              (ffa + ffa);

  ////// Condition: Boolean //////////////////////

  //Value: Double
  ffb <<= cond(sva < 0.0, 0.0)
              (false, 1.0)
              (2.0);
  svb <<= cond(sva < 0.0, 0.0)
              (false, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(sva < 0.0, 0.0)
              (false, sva)
              (2.0);
  svb <<= cond(sva < 0.0, 0.0)
              (false, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(sva < 0.0, 0.0)
              (false, sva + sva)
              (2.0);
  svb <<= cond(sva < 0.0, 0.0)
              (false, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(sva < 0.0, 0.0)
              (false, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(sva < 0.0, 0.0)
              (false, ffa + ffa)
              (2.0);

  ////// Condition: Single Value Expression //////

  //Value: Double
  ffb <<= cond(sva < 0.0, 0.0)
              (sva < 0.0, 1.0)
              (2.0);
  svb <<= cond(sva < 0.0, 0.0)
              (sva < 0.0, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(sva < 0.0, 0.0)
              (sva < 0.0, sva)
              (2.0);
  svb <<= cond(sva < 0.0, 0.0)
              (sva < 0.0, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(sva < 0.0, 0.0)
              (sva < 0.0, sva + sva)
              (2.0);
  svb <<= cond(sva < 0.0, 0.0)
              (sva < 0.0, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(sva < 0.0, 0.0)
              (sva < 0.0, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(sva < 0.0, 0.0)
              (sva < 0.0, ffa + ffa)
              (2.0);

  ////// Condition: Full Expression //////////////

  //Value: Double
  ffb <<= cond(sva < 0.0, 0.0)
              (ffa < 0.0, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(sva < 0.0, 0.0)
              (ffa < 0.0, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(sva < 0.0, 0.0)
              (ffa < 0.0, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(sva < 0.0, 0.0)
              (ffa < 0.0, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(sva < 0.0, 0.0)
              (ffa < 0.0, ffa + ffa)
              (2.0);

  ////// Condition: Mask /////////////////////////

  //Value: Double
  ffb <<= cond(sva < 0.0, 0.0)
              (mask, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(sva < 0.0, 0.0)
              (mask, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(sva < 0.0, 0.0)
              (mask, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(sva < 0.0, 0.0)
              (mask, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(sva < 0.0, 0.0)
              (mask, ffa + ffa)
              (2.0);

  ////////////////////////////////////////////////
  // Previous: Full //////////////////////////////
  ////////////////////////////////////////////////

  ////// Condition: Final ////////////////////////

  //Value: Double
  ffb <<= cond(ffa < 0.0, 0.0)
              (1.0);

  //Value: Single Value Field
  ffb <<= cond(ffa < 0.0, 0.0)
              (sva);

  //Value: Single Value Expression
  ffb <<= cond(ffa < 0.0, 0.0)
              (sva + sva);

  //Value: Full Field
  ffb <<= cond(ffa < 0.0, 0.0)
              (ffa);

  //Value: Full Expression
  ffb <<= cond(ffa < 0.0, 0.0)
              (ffa + ffa);

  ////// Condition: Boolean //////////////////////

  //Value: Double
  ffb <<= cond(ffa < 0.0, 0.0)
              (false, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(ffa < 0.0, 0.0)
              (false, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(ffa < 0.0, 0.0)
              (false, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(ffa < 0.0, 0.0)
              (false, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(ffa < 0.0, 0.0)
              (false, ffa + ffa)
              (2.0);

  ////// Condition: Single Value Expression //////

  //Value: Double
  ffb <<= cond(ffa < 0.0, 0.0)
              (sva < 0.0, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(ffa < 0.0, 0.0)
              (sva < 0.0, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(ffa < 0.0, 0.0)
              (sva < 0.0, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(ffa < 0.0, 0.0)
              (sva < 0.0, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(ffa < 0.0, 0.0)
              (sva < 0.0, ffa + ffa)
              (2.0);

  ////// Condition: Full Expression //////////////

  //Value: Double
  ffb <<= cond(ffa < 0.0, 0.0)
              (ffa < 0.0, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(ffa < 0.0, 0.0)
              (ffa < 0.0, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(ffa < 0.0, 0.0)
              (ffa < 0.0, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(ffa < 0.0, 0.0)
              (ffa < 0.0, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(ffa < 0.0, 0.0)
              (ffa < 0.0, ffa + ffa)
              (2.0);

  ////// Condition: Mask /////////////////////////

  //Value: Double
  ffb <<= cond(ffa < 0.0, 0.0)
              (mask, 1.0)
              (2.0);

  //Value: Single Value Field
  ffb <<= cond(ffa < 0.0, 0.0)
              (mask, sva)
              (2.0);

  //Value: Single Value Expression
  ffb <<= cond(ffa < 0.0, 0.0)
              (mask, sva + sva)
              (2.0);

  //Value: Full Field
  ffb <<= cond(ffa < 0.0, 0.0)
              (mask, ffa)
              (2.0);

  //Value: Full Expression
  ffb <<= cond(ffa < 0.0, 0.0)
              (mask, ffa + ffa)
              (2.0);

  return 0;
}
