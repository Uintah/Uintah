
#include "Parallel.h"
#include "SimulationController.h"
#include <Core/Exceptions/Exception.h>
#include <Core/Malloc/Allocator.h>
#include <ieeefp.h>
#include <iostream>
#include <mpi.h>

using namespace SCIRun;
using std::cerr;

int main(int argc, char** argv)
{
    Parallel::initializeManager(argc, argv);

    fpsetmask(FP_X_OFL|FP_X_DZ|FP_X_INV);
    cerr << "mask: " << fpgetmask() << '\n';

    SimulationController* sim = scinew SimulationController(argc, argv);
    try {
	sim->run();
    } catch (Exception& e) {
	cerr << "Caught exception: " << e.message() << '\n';
    }

    Parallel::finalizeManager();
}
