#include <testprograms/Component/framework/TestPortImpl.h>

#include <iostream>

namespace sci_cca {

using namespace std;

TestPortImpl::TestPortImpl()
{
}

TestPortImpl::~TestPortImpl()
{
}

void
TestPortImpl::test()
{
  cerr << "test port:: test\n";
}

} // namespace sci_cca
