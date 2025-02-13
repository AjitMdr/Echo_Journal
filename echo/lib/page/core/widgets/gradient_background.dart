import 'package:flutter/material.dart';
import '../constants/app_styles.dart';
import './curved_pattern.dart';

class GradientBackgroundWithPattern extends StatelessWidget {
  final Widget child;

  const GradientBackgroundWithPattern({
    super.key,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: AppStyles.gradientColors,
        ),
      ),
      child: Stack(
        children: [
          Positioned.fill(
            child: CustomPaint(
              painter: CurvedPatternPainter(),
            ),
          ),
          child,
        ],
      ),
    );
  }
}