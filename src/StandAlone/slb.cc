/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Grid/GridP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Math/Primes.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>


#include <iostream>
#include <fstream>
#include <cstring>

// TODO:
//  DONE Parse geometry from UPS file and fill in weights array properly
//  DONE Fix uniform division (40/5 = 8)
//  DONE Proper arg parsing (for -div and integer submultiples)
//  DONE Convert patches to world coordinate space and print out
//  DONE Modify UPS file and write it out
//  Bounds for pieces (faster)
//  Summed area table (faster)
//  Allow integer submultiples
// 3. Test with different problems (disks with different weights)
// 5. Benchmark on a real problem
// 4. give to jim
// these are little things
// 6. Recover properly if trying to bisect 1 cell.
// 7. See if ProblemSpec::get should return 'this'.
// 8. Make work for all boxes on multiple-box runs
// 9. Make extraCells be a level property instead of a Box property
// 10. Compensate for jobs with no particles (divide domain evenly)
// Zoltan

using namespace Uintah;
using namespace std;

void bisect(const string& div, int num, int factor, 
            Uintah::Primes::FactorType factors, const IntVector& low, 
            const IntVector& high, const Array3<float>& weights,
            const LevelP& level, ProblemSpecP& level_ups, IntVector& extraCells)
{
  static int levels_of_recursion = -1;

  int index = -1;

  // l is to bisect in the dimension of the largest amount of cells
  if(div[num] == 'l') { 
    int x = high.x() - low.x();
    int y = high.y() - low.y();
    int z = high.z() - low.z();
    if (x >= y && x >= z)
      index = 0;
    else if (y > x && y >= z)
      index = 1;
    else if (z > y && z > x)
      index = 2;
  }
  else if(div[num] == 'x')
    index=0;
  else if(div[num] == 'y')
    index=1;
  else if(div[num] == 'z')
    index=2;
  else {
    throw InternalError("bad bisection axis: "+div[num], __FILE__, __LINE__);
  }

  if(factor == -1){
    static int idx = 0;
    Point low_point = level->getNodePosition(low);
    Point high_point = level->getNodePosition(high);
    IntVector res = high - low;

    ProblemSpecP box = level_ups->appendChild("Box");
    box->appendElement("label", idx);
    box->appendElement("lower", low_point);
    box->appendElement("upper", high_point);
    
    if (extraCells != IntVector(0,0,0))
      box->appendElement("extraCells", extraCells);
    box->appendElement("resolution", res);

    cerr << idx++ << ": patch: " << low << "-" << high << ',' 
         << low_point << "-" << high_point << '\n';



  } else {

    levels_of_recursion++;
    for (int qq = 0; qq < levels_of_recursion*2; qq++) {
      cout << ' ';
    }
    cout << "Bisect: dir " << div[num] << " factor: " << factors[factor] << ' ' 
         << low << '-' << high << endl;

    int number = static_cast<int>(factors[factor]);  // quiet the sgi compiler

    int l = low[index];
    int h = high[index];

    float total = 0;

    vector<float> sums(h-l);
    for(int i = l; i<h; i++){
      IntVector slab_low = low;
      slab_low[index] = i;
      IntVector slab_high = high;
      slab_high[index] = i;
      double sum = 0;
      for(CellIterator iter(slab_low, slab_high); !iter.done(); iter++) {
        sum += weights[*iter];
        //cout << weights[*iter] << ' ';
      }
      total += sum;
      //cout << sum << ' ' << total << '\n';
      //cerr << "sum[" << i-l << ": " << slab_low << "-" << slab_high << ": " << sum << '\n';
      sums[i-l] = total;
    }
    //total = 0;
    //for(int i = l; i < h; i++){
    //total += sums[i-l];
    //  sums[i-l] = total;
    //}
    double weight_per = total/number;

    int next_s = -1;
    for(int i = 0 ; i < number; i ++){
      double w1 = weight_per * i;
      double w2 = weight_per * (i+1);
      //cout << weight_per << ' ' << w1 << ' ' << w2 << '\n';
      int s = 0;
      int e = 0;

      if (next_s == -1) {
        while(sums[s] < w1 && s < h-l) {
          s++;
        }
      }
      else
        s = next_s;

      // find the upper end, compensating for 0's.
      float high_sum = sums[e];
      while((sums[e] <= w2 || high_sum >= sums[e]) && e < h-l) {
        //cout << e << ' ' << high_sum << ' ' << sums[e] << '\n';
        high_sum = sums[e];
        e++;
      }
      IntVector new_low = low;
      new_low[index] = s+l;
      IntVector new_high = high;
      new_high[index] = e+l;
      next_s = e;

      bisect(div, num+1, factor-1, factors, new_low, new_high, weights, level, level_ups, extraCells);
    }
    levels_of_recursion--;
  }
}

void usage( char *prog_name );

void
parseArgs( int argc, char *argv[], 
           int & nump, float & weight, string & infile, string & outfile,
           string & divisions, int & submultiples)
{
  if( argc < 4 || argc > 9 ) {
    usage( argv[0] );
  }

  weight = atof( argv[1] );
  if( weight < -1.0 || weight > 1.0 ) {
    cerr << "Weight must be between 0.0 and 1.0\n";
    exit( 1 );
  }
  nump = atoi( argv[2] );
  if( nump < 1 ) {
    cerr << "Number of patches must be greater than 0.\n";
    exit( 1 );
  }

  infile = argv[3];
  outfile = argv[4];

  // parse optional arguments
  for (int i = 5; i < argc; i++) {
    if (strcmp(argv[i], "-div") == 0) {
      i++;
      if (i >= argc) {
        cerr << "-div option needs an argument (i.e., -div xyz)\n";
        exit( 1 );
      }
      for (unsigned int j = 0; j < strlen(argv[i]); j++) {
        if (argv[i][j] != 'x' && argv[i][j] != 'y' && argv[i][j] != 'z') {
          cerr << "Divisions string must be a combination of x, y, or z\n";
          exit( 1 );
        }
      }
      divisions = argv[i];
    }
    else if (strcmp(argv[i], "-sub") == 0) {
      i++;
      if (i >= argc) {
        cerr << "-sub option needs an argument (i.e., -sub 2)\n";
        exit( 1 );
      }
      submultiples = atoi(argv[i]);
      if (submultiples < 1) {
        cerr << "Number of Submultiples must be greater than 0\n";
        exit( 1 );
      }
    }
    else {
      cerr << "Unknown option " << argv[i] << '\n';
      exit( 1 );
    }
  }
}

void
usage( char *prog_name )
{
  cout << "Usage: " << prog_name << " weight num_patches infile outfile "
       << "[-div <pattern>] [-sub <submult>] \n";
  cout << "    weight: 0.0 - 1.0.  0.0 => all cells, 1.0 => all particles\n";
  cout << "    num_patches: number of patches to divvy up domain into.\n";
  cout << "    infile:      .ups file to read in.\n";
  cout << "    outfile:     modified .ups file.\n";
  cout << "    -div <pattern>:     OPTIONAL.  pattern is some combination\n"
       << "        of x,y, and z, where the default is xyz\n";
  cout << "    -sub <submult>:     divide original resolution by submult"
       << "        for quicker estimation\n";
  exit( 1 );
}

int
main(int argc, char *argv[])
{
  try {
    Uintah::Parallel::determineIfRunningUnderMPI( argc, argv );
    Uintah::Parallel::initializeManager( argc, argv );

    int    nump;
    float  weight;
    string infile;
    string outfile;
    int submultiples = 1;
    string divisions = "l";
    
    parseArgs( argc, argv, nump, weight, infile, outfile,
               divisions, submultiples );
    
    // Get the problem specification

    ProblemSpecP ups = ProblemSpecReader().readInputFile( infile );

    if( !ups ) {
      throw ProblemSetupException("Cannot read problem specification", __FILE__, __LINE__);
    }
    
    if( ups->getNodeName() != "Uintah_specification" ) {
      throw ProblemSetupException("Input file is not a Uintah specification", __FILE__, __LINE__);
    }
    
    const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();
    
    // Setup the initial grid
    GridP grid=scinew Grid();
    IntVector extraCells(0,0,0);

    // save and remove the extra cells before the problem setup
    ProblemSpecP g = ups->findBlock("Grid");
    for( ProblemSpecP levelspec = g->findBlock("Level"); levelspec != 0;
         levelspec = levelspec->findNextBlock("Level")) {
      for (ProblemSpecP box = levelspec->findBlock("Box"); box != 0 ; 
           box = box->findNextBlock("Box")) {
        
        ProblemSpecP cells = box->findBlock("extraCells");
        if (cells != 0) {
          box->get("extraCells", extraCells);
          box->removeChild(cells);
        }
      }
    }
      
    grid->problemSetup(ups, world, false);  
    
    for (int l = 0; l < grid->numLevels(); l++) {
      const LevelP &level = grid->getLevel(l);
      
      IntVector low, high;
      level->findCellIndexRange(low, high);
      IntVector diff = high-low;
      long cells = diff.x()*diff.y()*diff.z();
      if(cells != level->totalCells())
        throw ProblemSetupException("Currently slb can only handle square grids", __FILE__, __LINE__);

      Uintah::Primes::FactorType factors;
      int n = Uintah::Primes::factorize(nump, factors);
      cerr << nump << ": ";
      for(int i=0;i<n;i++){
        cerr << factors[i] << " ";
      }
      cerr << '\n';
      
      string div = divisions;
      while(static_cast<int>(div.length()) <= n)
        div += divisions;
      
      Array3<float> weights(low, high);
      weights.initialize(weight);
      
      // Parse the geometry from the UPS
      ProblemSpecP mp = ups->findBlockWithOutAttribute("MaterialProperties");
      ProblemSpecP mpm = mp->findBlock("MPM");
      for (ProblemSpecP child = mpm->findBlock("material"); child != 0;
           child = child->findNextBlock("material")) {
        for (ProblemSpecP geom_obj_ps = child->findBlock("geom_object");
             geom_obj_ps != 0;
             geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
          vector<GeometryPieceP> pieces;
          GeometryPieceFactory::create(geom_obj_ps, pieces);
          
          GeometryPieceP mainpiece;
          if(pieces.size() == 0){
            throw ProblemSetupException("No piece specified in geom_object", __FILE__, __LINE__);
          } else if(pieces.size() > 1){
            mainpiece = scinew UnionGeometryPiece(pieces);
          } else {
            mainpiece = pieces[0];
          }
          
          for(CellIterator iter(low, high); !iter.done(); iter++){
            Point p = level->getCellPosition(*iter);
            if(mainpiece->inside(p))
              weights[*iter] = 1;
          }
        }
      }

      //      cout << "FIRST OUTPUT OF CELLS" << low << '-' << high << '\n';
      //      int blah = 0;
//       for(CellIterator iter(low, high); !iter.done(); iter++){
//         blah++;
//         cout << weights[*iter] << ' ';
//         if (blah % 12 == 0)
//           cout << '\n';
//         if (blah % 144 == 0)
//           cout << '\n';
//       }
//       cout << "\n\n";
      
      int factor = n-1;
      
      // remove the 'Box' entry from the ups - note this should try to
      // remove *all* boxes from the level node
      ProblemSpecP lev = g->findBlock("Level");
      ProblemSpecP box = lev->findBlock("Box");
      
      lev->removeChild(box);
      
      bisect(div, 0, factor, factors, low, high, weights, level, lev, 
             extraCells); 
    }
    
    ofstream out(outfile.c_str());
    out << ups;
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << '\n';
    if(e.stackTrace())
      cerr << "Stack trace: " << e.stackTrace() << '\n';
  } catch(...){
    cerr << "Caught unknown exception\n";
  }
}
