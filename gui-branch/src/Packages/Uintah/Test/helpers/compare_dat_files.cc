#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

//bool getNextNumber(istream& in, double& number);
bool getNextNumber(char*& nptr, double& number);

int main(int argc, char** argv)
{
  int i;

  if (argc < 5) {
    cerr << "usage: " << argv[0] << " {abs error allowed} {rel error allowed} {uda directory 1} {uda directory 2} {dat file names}\n";
    return -10;
  }

  bool failed = false;
  char datfilename[2][1000];

  for (int argv_index = 5; argv_index < argc; argv_index++) {
    bool has_significant_error = false;
    cout << "Comparing " << argv[argv_index] << "... ";

    ifstream datfile[2];
    for (i = 0; i < 2; i++) {
      *datfilename[i] = '\0';
      strcpy(datfilename[i], argv[i+3]);
      strcat(datfilename[i], argv[argv_index]);
      //cout << endl << endl << datfilename[i] << endl << endl;
      datfile[i].open(datfilename[i]);
      if (datfile[i].fail()) {
        cerr << "Cannot open file '" << datfilename[i] << "'" << endl;
        return -10;
      }
     }

    double allowable_abs_error = atof(argv[1]);
    double allowable_rel_error = atof(argv[2]);
    double greatest_error_time[2];
    double first_significant_error_time[2];
    
    double d[2];
    double max_abs;
    
    double greatest_rel_error = 0;
    char buf0[500];
    char buf1[500];
    char* bufloc[2];
    double time[2];

    while (!datfile[0].eof() && !datfile[1].eof()) {
      datfile[0].getline(buf0, 500);
      datfile[1].getline(buf1, 500);
      if (datfile[0].fail()) break;
      if (datfile[1].fail()) break;
      bufloc[0] = buf0; bufloc[1] = buf1;
      bool firstnumber = true;
      while (getNextNumber(bufloc[0], d[0]) && getNextNumber(bufloc[1], d[1])){
	if (firstnumber) {
	  time[0] = d[0];
	  time[1] = d[1];
	  firstnumber = false;
	}
	max_abs = fabs(d[0]);
	if (fabs(d[1]) > max_abs)
	  max_abs = fabs(d[1]);
	
	if (max_abs > 0) { // otherwise, both are zero
	  if (fabs(d[1] - d[0]) > allowable_abs_error) {
	    double rel_error = fabs(d[1] - d[0]) / max_abs;
	    if (rel_error > greatest_rel_error) {
	      greatest_rel_error = rel_error;
	      greatest_error_time[0] = time[0];
	      greatest_error_time[1] = time[1];
	      if (!has_significant_error && rel_error > allowable_rel_error) {
		first_significant_error_time[0] = time[0];
		first_significant_error_time[1] = time[1];
		has_significant_error = true;
	      }
	    }
	  }
	}
      }
      
    }
    
    if (has_significant_error) {
      cout << "\n\tgreatest relative error: " << greatest_rel_error
	   << "\n\tat times: " << greatest_error_time[0] << " / "
	   << greatest_error_time[1]
   	   << "\n\tand first signifant error at: "
	   << first_significant_error_time[0] << " / "
	   << first_significant_error_time[1] << endl;
      cout << "\nThe following is the suggested command to compare these files:\n";

      cout << "xdiff\\\n" << datfilename[0] << "\\\n" << datfilename[1] << "\n\n";
      
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

bool getNextNumber(char*& nptr, double& number)
{
  char* endptr;
  if (*nptr == '\0') {
    return false; // end of string -- no number
  }
  number = strtod(nptr, &endptr);
  while (endptr == nptr) {
    // skip non-numerical characters
    nptr++;
    if (*nptr == '\0') {
      return false; // end of string -- no number
    }
    number = strtod(nptr, &endptr);
  }
  nptr = endptr;
  return true;
}

/*
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
*/
