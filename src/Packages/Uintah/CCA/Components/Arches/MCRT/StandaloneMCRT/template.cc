
   surfaceFlag = TOP;
    
    // consider Z[] array from 0 to Npz -1 = Ncz
    kIndex = Ncz-1; // iIndex, jIndex, kIndex is consitent with control volume's index
    
    for ( jIndex = 0; jIndex < Ncy; jIndex ++ ) {
      for ( iIndex = 0; iIndex < Ncx; iIndex ++){
	//	cout << "iIndex = " << iIndex <<"; jIndex = " << jIndex << endl;
	surfaceIndex = iIndex + jIndex * Ncx;
	thisRayNo = rayNo_surface[surfaceFlag][surfaceIndex];

	if ( thisRayNo != 0 ) { // rays emitted from this surface

	  TopRealSurafce obTop(iIndex, jIndex, kIndex, Ncx);
	  RealPointer = &obTop;


 
     } // end iIndex
   
  }// end jIndex
  
} // end if rayNoSurface ! = 0 ?
   

}
