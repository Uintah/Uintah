#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

bool getNextNumber(istream& in, double& number);

int main(int argc, char** argv)
{
  int i;

  if (argc < 5) {
    cerr << "usage: " << argv[0] << " {abs error allowed} {rel error allowed} {uda directory 1} {uda directory 2} {dat file names}\n";
    return -10;
  }

  bool failed = false;
  char datfilename[500];

  for (int argv_index = 5; argv_index < argc; argv_index++) {
    cout << "Comparing " << argv[argv_index] << "... ";

    ifstream datfile[2];
    for (i = 0; i < 2; i++) {
      *datfilename = '\0';
      strcpy(datfilename, argv[i+3]);
      strcat(datfilename, argv[argv_index]);
      datfile[i].open(datfilename);
      if (datfile[i].fail()) {
        cerr << "Cannot open file '" << datfilename[i] << "'" << endl;
        return -10;
      }
    }
    
    double allowable_abs_error = atof(argv[1]);
    double allowable_rel_error = atof(argv[2]);
    
    double d[2];
    double max_abs;
    
    double greatest_rel_error = 0;
    while (getNextNumber(datfile[0], d[0]) && getNextNumber(datfile[1], d[1]))
    {
      max_abs = fabs(d[0]);
      if (fabs(d[1]) > max_abs)
        d[1] = fabs(d[1]);

      if (max_abs > 0) // otherwise, both are zero
      {
	if (fabs(d[1] - d[0]) > allowable_abs_error) {
	  double rel_error = fabs(d[1] - d[0]) / max_abs;
	  if (rel_error > greatest_rel_error) {
	    greatest_rel_error = rel_error;
	  }
	}
      }
    }
    
    if (greatest_rel_error > allowable_rel_error) {
      cout << "\n\tgreatest relative error: " << greatest_rel_error
           << endl;

      failed = true;
    }
    else
      cout << "good\n";

    for (i = 0; i < 2; i++)
      datfile[i].close();
  }
   
  if (failed) {
    cout << "\nOne or more dat files are not within allowable error.\n";
    return 1;
  }
  else {
    cout << "\nDat files are all within allowable error.\n";
    return 0;
  }
}

bool getNextNumber(istream& in, double& number)
{
  if (ws(in).eof())
    return false;

  in >> number;
 
  char dummy;
  while (in.fail()) {
    in.clear();
    in >> dummy; // extract non-numeric character and continue
    if (ws(in).eof())
      return false;
    in >> number;
  }         
  return true;
}

