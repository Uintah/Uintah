
BEGIN {
	i=0;
        nl=0;
	}

{
	if ($3 == "")next;
	name=$3 " " $4 " " $5 " " $6;
	if (size[name] == 0){
	   names[i]=name;
	   i++;
        }
	size[name]+=$2;
	n[name]++;
	nl++;
	if(nl%5000 == 0){
		printf("%d lines processed\n", nl);
	}
}
END {
	t=0;
	for (ii=0;ii<i;ii++){
		printf("%s: %d bytes total (%d)\n", names[ii], size[names[ii]], n[names[ii]]);
		t+=size[names[ii]];
		tn+=n[names[ii]];
	}
        printf("Grand Total: %d (%d)\n", t,tn);
}
