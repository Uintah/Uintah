/*
 * Takes input NRRD or ND and adds properties parsed from a markers file.
 * Markers file is colon delimited with the name in field 1 and the time
 * index in field 2.
 *
 * Author: Alex Ade
 * Date:   23 January 2005
 */

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Ports/NrrdPort.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <map>
#include <sstream>

using namespace SCIRun;

vector<string> tokenize(string str, string delim) {
   vector<string> tokens;
   int pos, pos2;
   pos = str.find_first_not_of(delim, 0);
   
   while (pos >= 0) {
      pos2 = str.find_first_of(delim, pos);
   
      if (pos2 < 0)
         pos2 = str.length();
   
      tokens.push_back(str.substr(pos, pos2 - pos));
      pos = str.find_first_not_of(delim, pos2);
   }
   
   return tokens;
}

int
main(int argc, char **argv) {
  NrrdDataHandle handle(0);

  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " NRRDIn NRRDOut MarkersFile\n";
    return 0;
  }

  string infn(argv[1]);
  string outfn(argv[2]);
  string mkrfn(argv[3]);

  int len = infn.size();
  const string ext(".nd");

  if (infn.substr(len - 3, 3) == ext) {
    Piostream *stream = auto_istream(infn);

    if (!stream) {
      cerr << "Error: can't open file " << infn << "." << endl;
      exit(0);
    }

    Pio(*stream, handle);

    if (!handle.get_rep() || stream->error()) {
      cerr << "Error: can't read data from file " << infn << "." << endl;
      delete stream;
      exit(0);
    }
    delete stream;
  } else {
    NrrdData *n = scinew NrrdData;
    if (nrrdLoad(n->nrrd=nrrdNew(), strdup(infn.c_str()), 0)) {
      char *err = biffGetDone(NRRD);
      cerr << "Error: can't read data from file " << infn << "." << endl;
      free(err);
      exit(0);
    }
    handle = n;
  }

  ifstream mkrstr;
  mkrstr.open(mkrfn.c_str());

  if (!mkrstr) {
    cerr << "Error: can't open file " << mkrfn << "." << endl;
    exit(0);
  }

  char buf[512];
  vector<string> tokens;
  string key, value;
  //int val;
  if (mkrstr.is_open()) {
    while (!mkrstr.eof()) {
      mkrstr.getline(buf, 512);
      string a(buf);
      if (a.length() > 0) {
        tokens = tokenize(a, ":");
        key = tokens[0];
        value = tokens[1];

        //stringstream ss(value);
        //ss >> val;

        //handle->set_property(key, val, false);
        handle->set_property(key, value, false);
      }
      tokens.clear();
    }
  }

  mkrstr.close();

  Piostream *stream = scinew BinaryPiostream(outfn, Piostream::Write);

  if (stream->error()) {
    cerr << "Error: can't write file " << outfn << "." << endl;
  } else {
    Pio(*stream, handle);
    delete stream;
  }

  return 0;  
}
