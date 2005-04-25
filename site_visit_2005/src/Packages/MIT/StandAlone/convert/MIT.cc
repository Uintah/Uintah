/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  MIT.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Thread/Thread.h>
#include <Packages/MIT/Core/Datatypes/MetropolisData.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;
using std::cin;
using std::cout;

using namespace SCIRun;
using namespace MIT;

int
main(int argc, char **argv) 
{
  if (argc != 4) {
    cerr << "Usage: "<<argv[0]<<" input mesurement-file distribution-file\n";
    return 0;
  }
  
  FILE *file = fopen(argv[1],"r");
  if ( !file ) {
    cerr << "Can not open input file [" << argv[1] << "]\n";
    return 0;
  }

  MeasurementsHandle measurements =  new Measurements;
  DistributionHandle pd = new Distribution;

  int n;
  fscanf (file, "%d", &n );

  measurements->t.resize(n);
  for (int i=0; i<n; i++) 
    fscanf( file, " %lg", &measurements->t[i]);

  int c;
  fscanf (file, " %d", &c );
  
  measurements->concentration.newsize(c,n);

  for (int i=0; i<c; i++) 
    for (int j=0; j<n; j++ )
      fscanf( file, "%lg", &measurements->concentration(i,j) );

  TextPiostream out_stream(argv[2], Piostream::Write);
  Pio(out_stream, measurements);

  int m;
  fscanf( file, "%d", &m);

  pd->sigma.newsize(m,m);

  for (int i=0; i<m; i++)
    for (int j=0; j<m; j++ )
      fscanf( file, "%lg",  &pd->sigma(i,j));

  pd->theta.resize(n);
  for (int i=0; i<n; i++)
    fscanf( file, "%lg", &pd->theta[i]);

  pd->kappa = 0.05;

  TextPiostream pd_stream(argv[3], Piostream::Write);
  Pio(pd_stream, pd);

  return 0;  
}    
