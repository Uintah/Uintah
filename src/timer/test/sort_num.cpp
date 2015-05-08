#include <stdio.h>
#include <timer.hpp>
#include <algorithm>
extern int arr[1000000];
void sort_num()
{
    StatementTimer(std::sort(arr,arr+1000000));
}
