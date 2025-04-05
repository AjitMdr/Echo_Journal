import 'package:flutter/material.dart';
import 'app-styles.dart';
import 'curved-pattern.dart';

class GradientBackgroundWithPattern extends StatelessWidget {
  final Widget child;

  const GradientBackgroundWithPattern({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    // Determine whether it's dark mode or light mode
    final isDarkMode = Theme.of(context).brightness == Brightness.dark;

    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: AppStyles.gradientColors, // Your predefined gradient colors
        ),
      ),
      child: Stack(
        children: [
          Positioned.fill(
            child: CustomPaint(
              painter: CurvedPatternPainter(), // The pattern remains the same
            ),
          ),
          // Apply dark mode specific background color
          Container(
            color:
                isDarkMode ? Colors.black.withOpacity(0.7) : Colors.transparent,
            child: child,
          ),
        ],
      ),
    );
  }
}
