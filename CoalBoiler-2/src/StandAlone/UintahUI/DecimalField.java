//**************************************************************************
// Program : DecimalField.java
// Purpose : An extension of JTextField to take care of decimals
// Author  : Biswajit Banerjee
// Date    : 12/7/1998
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.text.Format;
import java.text.DecimalFormat;
import java.text.ParseException;
import javax.swing.*;
import javax.swing.text.*;

//**************************************************************************
// Class   : DecimalField
// Purpose : Creates a text field that validates real numbers
//**************************************************************************
public class DecimalField extends JTextField {

  // Data
  protected DecimalFormat formatter;

  // Data that may be needed later

  // Constructor
  public DecimalField(double value, int columns) {
    
    // Set the size of the component
    super(columns);
    
    // Get the number formatter
    formatter = new DecimalFormat();
    formatter.applyPattern(new String("#0.0#"));
    setDocument(new RealNumberDocument(formatter));
    setValue(value);
  }

  public DecimalField(double value, int columns, boolean exp) {
    
    // Set the size of the component
    super(columns);
    
    // Get the number formatter
    String pattern = "#0.0#";
    String patternExp = "0.0##E0";
    if (exp) {
      formatter = new DecimalFormat(patternExp);
    } else {
      formatter = new DecimalFormat(pattern);
    }
    setDocument(new RealNumberDocument(formatter));
    setValue(value);
  }

  // Get method
  public double getValue() {
    double retVal = 0.0;
    try {
      String input = getText().toUpperCase();
      retVal = formatter.parse(input).doubleValue();
    } catch (ParseException e) {
    }
    return retVal;
  }

  // set method
  public void setValue(double value) {
    try {
      String text = formatter.format(value);
      setText(text);
    } catch (IllegalArgumentException e) {
      System.out.println("Cannot format "+value);
    }
  }

  // set method
  public void setValue(String value) {
    setText(value);
  }

  // Create the related document
  protected Document createDefaultModel() {
    return new RealNumberDocument(formatter);
  }

  // Inner class for whole number document
  protected class RealNumberDocument extends PlainDocument {

    private Format format;

    // constructor
    public RealNumberDocument(Format f) {
      format = f;
    }

    // The insert string method
    public void insertString(int offs, String str, AttributeSet a)
      throws BadLocationException {

        String currentText = getText(0, getLength());
        String beforeOffset = currentText.substring(0, offs);
        String afterOffset = currentText.substring(offs, currentText.length());
        String proposedResult = beforeOffset + str + afterOffset;

        try {
            format.parseObject(proposedResult);
            super.insertString(offs, str, a);
        } catch (ParseException e) { }
    }

    // The remove method
    public void remove(int offs, int len) throws BadLocationException {
        String currentText = getText(0, getLength());
        String beforeOffset = currentText.substring(0, offs);
        String afterOffset = currentText.substring(len + offs, 
                             currentText.length());
        String proposedResult = beforeOffset + afterOffset;

        try {
            if (proposedResult.length() != 0)
                format.parseObject(proposedResult);
            super.remove(offs, len);
	} catch (ParseException e) { }
    }    

  }
}
