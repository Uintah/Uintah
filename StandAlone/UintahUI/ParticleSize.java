/**************************************************************************
// Program : ParticleSize.java
// Purpose : Create a ParticleSize object that stores the particle size
//           inputs and calculated data on particle sizes.
// Author  : Biswajit Banerjee
// Date    : 05/05/2006
// Mods    :
//**************************************************************************/

//************ IMPORTS **************
import java.text.DecimalFormat;

//**************************************************************************
// Class   : ParticleSize
// Purpose : The ParticleSize data structure contains
//           1) String compositeName
//           2) double volFracinComposite
//           3) double sizeInp[]
//           4) double volFracInp[]
//           5) int nofSizeFractions
//           6) double sizeFractionsCalc[]
//           7) double frequency2DCalc[]
//           8) double frequency3DCalc[]
//           9) double volumeFraction2DCalc[]
//          10) double volumeFraction3DCalc[]
//**************************************************************************
public class ParticleSize {

  // Static variables and constants
  public static final int NOF_SIZES = 100;

  // Data
  public String compositeName = null;
  public double volFracInComposite = 0.0;
  public int nofSizesInp = 0;
  public double[] sizeInp;
  public double[] volFracInp;
  public int nofSizesCalc = 0;
  public double[] sizeCalc;
  public int[] freq2DCalc;
  public int[] freq3DCalc;
  public double[] volFrac2DCalc;
  public double[] volFrac3DCalc;

  // Constructor
  public ParticleSize() {
    compositeName = "Default";
    volFracInComposite = 100.0;

    nofSizesInp = 2;
    sizeInp = new double[NOF_SIZES];
    volFracInp = new double[NOF_SIZES];

    nofSizesCalc = 2;
    sizeCalc = new double[NOF_SIZES];
    freq2DCalc = new int[NOF_SIZES];
    freq3DCalc = new int[NOF_SIZES];
    volFrac2DCalc = new double[NOF_SIZES];
    volFrac3DCalc = new double[NOF_SIZES];

    for (int ii = 0; ii < NOF_SIZES; ii++) {
      sizeInp[ii] = 0.0;
      volFracInp[ii] = 0.0;
      sizeCalc[ii] = 0.0;
      freq2DCalc[ii] = 0;
      freq3DCalc[ii] = 0;
      volFrac2DCalc[ii] = 0;
      volFrac3DCalc[ii] = 0;
    }

    sizeInp[0] = 100.0;
    volFracInp[0] = 10.0;
    sizeCalc[0] = 100.0;
    freq2DCalc[0] = 10;
    freq3DCalc[0] = 10;
    volFrac2DCalc[0] = 0.001;
    volFrac3DCalc[0] = 0.001;

    sizeInp[1] = 1000.0;
    volFracInp[1] = 90.0;
    sizeCalc[1] = 1000.0;
    freq2DCalc[1] = 90;
    freq3DCalc[1] = 90;
    volFrac2DCalc[1] = 0.999;
    volFrac3DCalc[1] = 0.999;
  }

  // Copy Constructor
  public ParticleSize(ParticleSize partSizeDist) {
    compositeName = partSizeDist.compositeName;
    volFracInComposite = partSizeDist.volFracInComposite;

    nofSizesInp = partSizeDist.nofSizesInp;;
    sizeInp = new double[NOF_SIZES];
    volFracInp = new double[NOF_SIZES];

    for (int ii = 0; ii < nofSizesInp; ii++) {
      sizeInp[ii] = partSizeDist.sizeInp[ii];
      volFracInp[ii] = partSizeDist.volFracInp[ii];
    }

    nofSizesCalc = partSizeDist.nofSizesCalc;
    sizeCalc = new double[NOF_SIZES];
    freq2DCalc = new int[NOF_SIZES];
    freq3DCalc = new int[NOF_SIZES];
    volFrac2DCalc = new double[NOF_SIZES];
    volFrac3DCalc = new double[NOF_SIZES];

    for (int ii = 0; ii < nofSizesCalc; ii++) {
      sizeCalc[ii] = partSizeDist.sizeCalc[ii];
      freq2DCalc[ii] = partSizeDist.freq2DCalc[ii];
      freq3DCalc[ii] = partSizeDist.freq3DCalc[ii];
      volFrac2DCalc[ii] = partSizeDist.volFrac2DCalc[ii];
      volFrac3DCalc[ii] = partSizeDist.volFrac3DCalc[ii];
    }
  }

  // Copy 
  public void copy(ParticleSize partSizeDist) {
    compositeName = partSizeDist.compositeName;
    volFracInComposite = partSizeDist.volFracInComposite;

    nofSizesInp = partSizeDist.nofSizesInp;;
    for (int ii = 0; ii < nofSizesInp; ii++) {
      sizeInp[ii] = partSizeDist.sizeInp[ii];
      volFracInp[ii] = partSizeDist.volFracInp[ii];
    }

    nofSizesCalc = partSizeDist.nofSizesCalc;
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      sizeCalc[ii] = partSizeDist.sizeCalc[ii];
      freq2DCalc[ii] = partSizeDist.freq2DCalc[ii];
      freq3DCalc[ii] = partSizeDist.freq3DCalc[ii];
      volFrac2DCalc[ii] = partSizeDist.volFrac2DCalc[ii];
      volFrac3DCalc[ii] = partSizeDist.volFrac3DCalc[ii];
    }
  }

  // Print
  public void print() {

    DecimalFormat df = new DecimalFormat("##0.00");
    System.out.println("Input");
    System.out.println("Size ... Vol.Frac");
    for (int ii = 0; ii < nofSizesInp; ii++) {
      System.out.println(df.format(sizeInp[ii])+"    "+
                         df.format(volFracInp[ii]));
    }
    System.out.println("Calculated");
    System.out.println("Size "+
                       "... Number (2D) .. Vol.Frac (2D)"+
                       "... Number (3D) .. Vol.Frac (3D)");
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      System.out.println(df.format(sizeCalc[ii])+"    "+
                         freq2DCalc[ii]+"     "+ 
                         df.format(volFrac2DCalc[ii])+"      "+
                         freq3DCalc[ii]+"     "+ 
                         df.format(volFrac3DCalc[ii]));
    }

  }
}
