#include <fstream>
#include <iostream>
#include <string>
#include "DaVinci.h"
#include "TaskGraph.h"

using namespace std;

int
main(int argc, char* argv[])
{
    if (argc < 2) {
    	cerr << "usage: " << argv[0] << " <uda directory>" << endl;
	return 1;
    }
    string infile = string(argv[1]) + "/taskgraph.xml";

    TaskGraph* graph = TaskGraph::inflate(infile);
    if (!graph) {
    	cerr << "Failed reading task graph, quitting" << endl;
    	return 1;
    }

    DaVinci* davinci = DaVinci::run();
    davinci->setOrientation(DaVinci::BOTTOM_UP);
    davinci->setGraph(graph);

    char buf[BUFSIZ + 1];
    ssize_t len = read(davinci->getOutput(), buf, BUFSIZ);
    buf[len] = '\0';
    cout << "main: " << buf << flush;

    delete davinci;
    delete graph;

    return 0;
}
