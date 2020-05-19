/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <StandAlone/tools/puda/gridStats.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>

#include <iomanip>
#include <vector>

using namespace Uintah;
using namespace std;

//______________________________________________________________________
//
void
Uintah::gridstats( DataArchive* da, CommandLineFlags & clf )
{
  vector<int> index;
  vector<double> times;
  da->queryTimesteps( index, times );
  ASSERTEQ(index.size(), times.size());

  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);

  for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){

    double time = times[t];
    cout << "__________________________________\n";
    cout << "Timestep " << t << ": " << time << "\n";
    GridP grid = da->queryGrid( t );
    grid->performConsistencyCheck();
    grid->printStatistics();

    Vector domainLength;
    grid->getLength(domainLength, "minusExtraCells");
    cout << "Domain Length:        " << domainLength << "\n";

    BBox box;
    grid->getInteriorSpatialRange( box );
    cout << "\nInterior Spatial Range: " << box << "\n";

    grid->getSpatialRange( box );
    cout << "Spatial Range:          " << box << "\n\n";

    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);
      cout << "Level: index " << level->getIndex() << ", id " << level->getID() << "\n";

      IntVector rr(level->getRefinementRatio());
      cout << "       refinement ratio: " << rr << "\n";

      BBox lbox;
      level->getInteriorSpatialRange( lbox );
      cout << "       Interior Spatial Range: " << lbox << "\n";

      level->getSpatialRange( lbox );
      cout << "       Spatial Range:          " << lbox << "\n\n";

      IntVector lo, hi;
      level->findInteriorCellIndexRange(lo,hi);
      cout << "Total Number of Cells:" << hi-lo << "\n";
      cout << "dx:                   " << level->dCell() << "\n";

      for( auto iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        
        IntVector lo = patch->getExtraNodeLowIndex();
        IntVector hi = patch->getExtraNodeHighIndex();
        
        Point loNode = patch->getNodePosition( lo );
        Point hiNode = patch->getNodePosition( hi );
        
        cout << *patch << "\n"
             << "\t   Spatial extents (including extra cells): " << loNode << " " << hiNode << "\n"
             << "\t   BC types: x- " << patch->getBCType(Patch::xminus) << ", x+ "<<patch->getBCType(Patch::xplus)
             << ", y- "<< patch->getBCType(Patch::yminus) << ", y+ "<< patch->getBCType(Patch::yplus)
             << ", z- "<< patch->getBCType(Patch::zminus) << ", z+ "<< patch->getBCType(Patch::zplus) << "\n";
      }
    }
  }
} // end gridstats()
