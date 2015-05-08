#include <stdio.h>
#include <timer.hpp>
#include <algorithm>
using namespace std;
int arr[1000000];
void gen_num();
void sort_num();
void print_num();
int main()
{
    StatementTimer(gen_num(););
    StatementTimer(sort_num(););
    //StatementTimer(print_num(););
    return 0;
}
