#include <tabprops/StateTable.h>

#include <vector>
#include <string>
#include <iostream>

int main()
{
  StateTable tbl;
  tbl.add_metadata("my double", 1.234);
  tbl.add_metadata("my int",    1234 );

  tbl.add_metadata("my string 1", std::string("hi"));
  tbl.add_metadata("my string 2", std::string("hello"));

  tbl.write_table("metadata.tbl");

  bool ok = true;

  // read the table
  StateTable tbl2;
  tbl2.read_table("metadata.tbl");
  ok = ok ? tbl2.get_metadata<double>("my double") == 1.234 : false;
  ok = ok ? tbl2.get_metadata<int>("my int") == 1234 : false;

  ok = ok ? tbl2.get_metadata<std::string>("my string 1") == "hi" : false;
  ok = ok ? tbl2.get_metadata<std::string>("my string 2") == "hello" : false;

  if( ok ){
    std::cout << "PASS" << std::endl;
    return 0;
  }

  std::cout << "FAIL" << std::endl;
  return -1;
}
