
void ssmult(int beg, int end, int* rows, int* columns,
	    double* a, double* xp, double* bp)
{
    int i, j;
    for(i=beg;i<end;i++){
	double sum=0;
	int row_idx=rows[i];
	int next_idx=rows[i+1];
	for(j=row_idx;j<next_idx;j++){
	    sum+=a[j]*xp[columns[j]];
	}
	bp[i]=sum;
   
    }
}
