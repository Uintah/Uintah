
#include "Parallel.h"
#include "SimulationController.h"
#include <SCICore/Exceptions/Exception.h>
using SCICore::Exceptions::Exception;
#include <ieeefp.h>
#include <iostream>
using std::cerr;
#include <mpi.h>
#include <SCICore/Malloc/Allocator.h>

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
