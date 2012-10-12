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

//
//  radius_maker.cc
//
//  Reads in a two dimensional (variables,particles) NRRD, and adds another 
//  variable (radius) based on a specified variable.  This is useful to, say,
//  use 'mass' to generated radii of particles.
//
//  Author: James Bigler
//  
//  Modified by: J. Davison de St. Germain
//
//  Date: Oct 18, 2007
//
//  Copyright (C) 2007 - C-SAFE - University of Utah
//

#include <teem/nrrd.h>
#include <cstdio>

#include <vector>
#include <string>

using namespace std;

float base_radius = 0.0004;

class Exception {
public:
  Exception(int index, float min, float max, float val):
    index(index), min(min), max(max), val(val)
  {}

  int index;
  float min;
  float max;
  float val;

  bool fits(float* data) {
//     fprintf(stderr, "min = %g, data[%d] = %g, max = %g\n", min, index, data[index], max);
    return (min <= data[index] && data[index] <= max);
  }
};

char * prog_name;

void
usage()
{
  printf( "\n"
          "%s: \n"
          "\n"
          "This program takes in a particle dataset in the form of a nrrd \n"
          "and creates a new nrrd adding a radius variable based on the mass.\n"
          "The format of the data should be 2 dimensional with the first dimension \n"
          "as the variables and the second as the particle.\n\n", prog_name );
  
  printf( "-i <file>    input nrrd (defaults to stdin)\n");
  printf( "-o <file>    output nrrd (defaults to stdout)\n");
  printf( "-r <float>   radius to normalize to (defaults to %g)\n", base_radius);
  printf( "-var <int> <float>  index of variable and value to use for normalization\n");
  printf( "-except <index> <min> <max> <norm> If variable is between the min and max, normalize\n"
          "                                   by norm instead of the one specified by -var\n");
  printf( "-help        prints this message\n");
  printf( "\n" );
  exit( 1 );
}

int
main( int argc, char *argv[] )
{
  char            * err;
  char            * in = const_cast<char*>("-");
  char            * out_file_name = const_cast<char*>("-");
  int               var = -1;
  float             global_norm = 1;
  vector<Exception> exceptions;

  prog_name = argv[0];

  if( argc == 1 ) {
    usage();
  }

  for(int i = 1; i < argc; i++) {

    string arg = argv[i];

    if ( arg == "-i" )  {
      in = argv[++i];
    } 
    else if ( arg == "-o" )  {
      out_file_name = argv[++i];
    }
    else if ( arg == "-r" )  {
      base_radius = atof(argv[++i]);
    }
    else if ( arg == "-var" )  {
      var = atoi(argv[++i]);
      global_norm = atof(argv[++i]);
    }
    else if ( arg == "-except")  {
      int   index = atoi(argv[++i]);
      float min   = atof(argv[++i]);
      float max   = atof(argv[++i]);
      float exp   = atof(argv[++i]);
      exceptions.push_back(Exception(index, min, max, exp));
    }
    else if(  arg == "-help" || arg == "--help" || arg == "-h" ) {
      usage();
    }
    else {
      fprintf(stderr, "%s: unrecognized option %s\n", prog_name, argv[i]);
      usage();
    }
  }

  if( var < 0 ) {
    printf( "%s: 'var' must be positive\n", prog_name );
    exit(2);
  }

  Nrrd *nin = nrrdNew();

  if( nrrdLoad(nin, in, 0) ) {
    err = biffGet(NRRD);
    printf( "%s: error loading nrrd : %s\n%s", prog_name, in, err );
    exit(2);
  }

  if( nin->dim != 2 ) {
    printf( "%s: %s does not have dims == 2\n", prog_name, in );
    exit(2);
  }
  
  if( (int)(nin->axis[0].size) <= var ) {
    printf( "%s: var (%d) is too large, %s->axis[0].size = %d\n", prog_name, var, in, (int)nin->axis[0].size );
    exit(2);
  }

  if (nin->type != nrrdTypeFloat) {
    printf( "%s: %s is not a float type\n", prog_name, in);
    exit(2);
  }


  int num_spheres = nin->axis[1].size;
  Nrrd * nout = nrrdNew();
  // Make sure to leave room for the new radius variable
  if( nrrdAlloc_va( nout, nrrdTypeFloat, nin->dim, nin->axis[0].size+1, nin->axis[1].size ) ) {
    err = biffGet(NRRD);
    fprintf(stderr, "%s: error allocating output nrrd\n%s", prog_name, err);
    exit(1);
  }

  printf("here: %d, %d\n",(int) nout->axis[0].size, (int)nout->axis[1].size );

  // Now copy over the data
  float * indata   = (float*)(nin->data);
  float * outdata  = (float*)(nout->data);
  int     num_vars = nin->axis[0].size;

  double  min = DBL_MAX;
  double  max = DBL_MIN;

  const int NUM_BINS = 40;
  int       bins[ NUM_BINS ];

  for( int cnt = 0; cnt < NUM_BINS; cnt++ ) { bins[ cnt ] = 0; }

  for( int i = 0; i < num_spheres; i++, indata += num_vars, outdata += num_vars+1 ) {

    for( int j = 0; j < num_vars; j++ ) {
      outdata[j] = indata[j];    // Copy the data over
    }

    float normal = global_norm;

    for(size_t e = 0; e < exceptions.size(); e++)
      if (exceptions[e].fits(indata)) {
        normal = exceptions[e].val;
        //        fprintf(stderr, "normal = %g\n", normal);
        break;
      }

    double value = indata[var];
    if( value < min ) min = value;
    if( value > max ) max = value;


    double base = 0.0000005;
    for( int cnt = 1; cnt <= NUM_BINS; cnt++ ) { 
      if( value > base ) {
        bins[ cnt-1 ]++;
        outdata[ num_vars ] = base_radius * ( (10 - (cnt-1))/ 10.0 );
        break;
      }
      base /= 4;
    }
    if( value < base ) {
      printf( "ERROR: This should not happen... variable is way too small...\n" );
      exit( 2 );
    }


#if 0
    float powresult = pow( value/normal, 0.33333333f );
    //    fprintf(stderr, "powresult = %g\n", powresult);
    outdata[num_vars] = base_radius * powresult;

    if (i%1000000 == 0) {
      printf( "indatad[%d] = %g, normal = %g, powresult = %g\n", var, value, normal, powresult );
      for(int j = 0; j < num_vars; j++) {
        printf( "indata[%d][%d] = %g\toutdata = %g\n", i, j, indata[j], outdata[j]);
      }
      printf( "radius outdata = %g\n", outdata[num_vars]);
    }
#endif
    
  }
  
  for( int cnt = 0; cnt < NUM_BINS; cnt++ ) { 
    if( ( bins[cnt] > 0 ) || cnt == (NUM_BINS-1) ) {
      printf( "\nbin[%d] = %d", cnt, bins[cnt] );
    } else {
      printf( "." ); fflush( stdout );
    }
  }

  printf( "\nmin/max: %.18lf, %.18lf\n", min, max );

  NrrdIoState * nios = nrrdIoStateNew();
  nrrdIoStateSet( nios, nrrdIoStateDetachedHeader, true );

  // Save the nrrd
  if( nrrdSave(out_file_name, nout, nios) ) {
    err = biffGet(NRRD);
    printf( "%s: error saving nrrd : '%s'\n\n%s\n\n", prog_name, out_file_name, err );
    exit(2);
  }
  return 0;
}
