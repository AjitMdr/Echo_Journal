import 'package:flutter/material.dart';

class AppTheme {
  static final ThemeData lightTheme = ThemeData(
    primaryColor: Color(0xFFff6f61), // Soft red
    scaffoldBackgroundColor: Colors.white,
    appBarTheme: AppBarTheme(
      backgroundColor: Color(0xFFF0F0F0), // Light Gray
      elevation: 2,
      titleTextStyle: TextStyle(
        color: Color(0xFF333333), // Dark Gray
        fontWeight: FontWeight.bold,
        fontSize: 24,
      ),
      iconTheme: IconThemeData(color: Color(0xFF555555)), // Dark Gray
    ),
    iconTheme: IconThemeData(color: Color(0xFF555555)), // Dark Gray
    buttonTheme: ButtonThemeData(
      buttonColor: Color(0xFFff6f61), // Soft red
      textTheme: ButtonTextTheme.primary,
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        foregroundColor: Colors.white,
        backgroundColor: Color(0xFFff6f61), // Soft red
        elevation: 5,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: Color(0xFFff6f61), // Soft red
        textStyle: TextStyle(fontWeight: FontWeight.bold),
      ),
    ),
    textTheme: TextTheme(
      bodyLarge: TextStyle(color: Color(0xFF333333)), // Dark Gray
      bodyMedium: TextStyle(color: Color(0xFF333333)), // Dark Gray
      titleLarge: TextStyle(
        color: Color(0xFF0A3954), // Dark Blue
        fontSize: 20,
        fontWeight: FontWeight.w600,
      ),
      titleMedium: TextStyle(
        color: Color(0xFF333333), // Dark Gray
        fontSize: 16,
        fontWeight: FontWeight.w400,
      ),
    ),
    cardTheme: CardTheme(
      color: Color(0xFFF7F7F7), // Light Gray
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
    ),
  );

  static final ThemeData darkTheme = ThemeData(
    primaryColor: Color(0xFFd32f2f), // Red Accent
    scaffoldBackgroundColor: Color(0xFF212121), // Dark Gray
    appBarTheme: AppBarTheme(
      backgroundColor: Color(0xFF212121), // Dark Gray
      elevation: 2,
      titleTextStyle: TextStyle(
        color: Color(0xFFd32f2f), // Red Accent
        fontWeight: FontWeight.bold,
        fontSize: 24,
      ),
      iconTheme: IconThemeData(color: Colors.white),
    ),
    iconTheme: IconThemeData(color: Colors.white70),
    buttonTheme: ButtonThemeData(
      buttonColor: Color(0xFFd32f2f), // Red Accent
      textTheme: ButtonTextTheme.primary,
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        foregroundColor: Colors.white,
        backgroundColor: Color(0xFFd32f2f), // Red Accent
        elevation: 5,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: Color(0xFFd32f2f), // Red Accent
        textStyle: TextStyle(fontWeight: FontWeight.bold),
      ),
    ),
    textTheme: TextTheme(
      bodyLarge: TextStyle(color: Colors.white70),
      bodyMedium: TextStyle(color: Colors.white54),
      titleLarge: TextStyle(
        color: Color(0xFFB0B0B0), // Light Gray
        fontSize: 20,
        fontWeight: FontWeight.w600,
      ),
      titleMedium: TextStyle(
        color: Color(0xFFB0B0B0), // Light Gray
        fontSize: 16,
        fontWeight: FontWeight.w400,
      ),
    ),
    cardTheme: CardTheme(
      color: Color(0xFF121212), // Very Dark Gray
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
    ),
  );
}
