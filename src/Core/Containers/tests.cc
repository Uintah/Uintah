//static char *id="@(#) $Id$";

#include <Tester/TestTable.h>
#include <Containers/String.h>
#include <Containers/Queue.h>
#include <Containers/Array1.h>
#include <Containers/Array2.h>
#include <Containers/Array3.h>
#include <Containers/HashTable.h>
#include <Containers/FastHashTable.h>
#include <Containers/BitArray1.h>
#include <Containers/PQueue.h>

namespace SCICore {
namespace Containers {

using SCICore::Tester::TestTable;

TestTable test_table[] = {
    {"clString", &clString::test_rigorous, &clString::test_performance},
    {"Queue", Queue<int>::test_rigorous, 0},
    {"Array1", Array1<float>::test_rigorous, 0},
    {"Array2", Array2<int>::test_rigorous, 0},
    {"Array3", Array3<int>::test_rigorous, 0},
//  {"HashTable", HashTable<char*, int>::test_rigorous, 0},
    {"FastHashTable", FastHashTable<int>::test_rigorous, 0},
    {"BitArray1", &BitArray1::test_rigorous, 0},
    {"PQueue", &PQueue::test_rigorous, 0},
    {0,0,0}

};

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:15  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

