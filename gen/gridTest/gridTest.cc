#include <iostream.h>
#include <Classlib/Array1.h>

main() {
    int dist=4, imax, jmax, kmax, i, j, k;
    cout << "Distance? ";
    cin >> dist;
    while (dist >= 0) {
	cout << "imax, jmax, kmax? ";
	cin >> imax >> jmax >> kmax;
	cout << "i, j, k? ";
	cin >> i >> j >> k;
	Array1<int>* set=new Array1<int>(0,10,12*(2*dist+1)*(2*dist+1));
	if (dist==0) {
	    set->add(i); set->add(j); set->add(k);
	} else {
	    int curri=i-dist, currj, currk;
	    for (curri=i-dist; curri<=i+dist; curri+=2*dist)
		if (curri>=0 && curri<=imax)
		    for (currj=j-dist; currj<=j+dist; currj++)
			if (currj>=0 && currj<=jmax)
			    for (currk=k-dist; currk<=k+dist; currk++)
				if (currk>=0 && currk<=kmax) {
				    set->add(curri); set->add(currj); set->add(currk);
				}
	    for (currj=j-dist; currj<=j+dist; currj+=2*dist)
		if (currj>=0 && currj<=jmax)
		    for (currk=k-dist; currk<=k+dist; currk++)
			if (currk>=0 && currk<=kmax)
			    for (curri=i-dist+1; curri<=i+dist-1; curri++)
				if (curri>=0 && curri<=imax)  {
				    set->add(curri); set->add(currj); set->add(currk);
				}
	    for (currk=k-dist; currk<=k+dist; currk+=2*dist)
		if (currk>=0 && currk<=kmax)
		    for (curri=i-dist+1; curri<=i+dist-1; curri++)
			if (curri>=0 && curri<=imax)
			    for (currj=j-dist+1; currj<=j+dist-1; currj++)
				if (currj>=0 && currj<=jmax) {
				    set->add(curri); set->add(currj); set->add(currk);
				}
	}
	for (int p=0; p<set->size(); p+=3) {
	    cout << (p+1)/3 << ": " << (*set)[p] << " " << (*set)[p+1] << " " << (*set)[p+2] << "\n";
	}
	delete set;
    }
}
